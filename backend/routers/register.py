from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import List
import json
from datetime import datetime

from models.student import StudentCreate, Student, StudentResponse, EmergencyContact, MedicalInfo
from services.face_engine import face_engine
from db.mongodb import get_students_collection

router = APIRouter()

@router.post("/register_student", response_model=dict)
async def register_student(
    student_data: str = Form(..., description="Student information as JSON string"),
    photos: List[UploadFile] = File(..., description="4-5 face photos of the student")
):
    """Register a new student with face photos"""
    
    try:
        # Parse student data from JSON string
        student_info = json.loads(student_data)
        
        # Validate minimum photos requirement
        if len(photos) < 3:
            raise HTTPException(status_code=400, detail="At least 3 photos are required")
        if len(photos) > 6:
            raise HTTPException(status_code=400, detail="Maximum 6 photos allowed")
        
        # Validate file types
        allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        for photo in photos:
            if photo.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type {photo.content_type}. Allowed: JPEG, PNG, WebP"
                )
        
        # Process photos and extract embeddings
        embeddings = []
        photo_data = []
        
        for i, photo in enumerate(photos):
            photo_bytes = await photo.read()
            photo_data.append(photo_bytes)
            
            # Validate face quality
            is_valid, message = face_engine.validate_face_quality(photo_bytes)
            if not is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Photo {i+1} quality issue: {message}"
                )
        
        # Extract embeddings from all photos
        embeddings = face_engine.get_multiple_embeddings(photo_data)
        
        if len(embeddings) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Failed to extract face embeddings from at least 2 photos. Please ensure faces are clearly visible."
            )
        
        # Create student object
        try:
            student_create = StudentCreate(**student_info)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid student data: {e}")
        
        # Create student document for database
        student_doc = {
            "name": student_create.name,
            "age": student_create.age,
            "gender": student_create.gender,
            "school": student_create.school,
            "grade": student_create.grade,
            "emergency_contact": student_create.emergency_contact.dict(),
            "medical_info": student_create.medical_info.dict() if student_create.medical_info else None,
            "notes": student_create.notes,
            "embeddings": embeddings,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Save to database
        collection = await get_students_collection()
        result = await collection.insert_one(student_doc)
        
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to save student to database")
        
        return {
            "success": True,
            "message": "Student registered successfully",
            "student_id": str(result.inserted_id),
            "embeddings_count": len(embeddings),
            "photos_processed": len(photos)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/students", response_model=List[StudentResponse])
async def get_all_students():
    """Get list of all registered students"""
    try:
        collection = await get_students_collection()
        
        # Get all active students, excluding embeddings for performance
        cursor = collection.find(
            {"is_active": True}, 
            {"embeddings": 0}  # Exclude embeddings from response
        )
        
        students = []
        async for student_doc in cursor:
            student_doc["id"] = str(student_doc["_id"])
            students.append(StudentResponse(**student_doc))
        
        return students
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")

@router.get("/student/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    """Get details of a specific student"""
    try:
        from bson import ObjectId
        
        collection = await get_students_collection()
        student_doc = await collection.find_one(
            {"_id": ObjectId(student_id), "is_active": True},
            {"embeddings": 0}  # Exclude embeddings
        )
        
        if not student_doc:
            raise HTTPException(status_code=404, detail="Student not found")
        
        student_doc["id"] = str(student_doc["_id"])
        return StudentResponse(**student_doc)
        
    except Exception as e:
        if "not found" in str(e):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to fetch student: {str(e)}")

@router.delete("/student/{student_id}")
async def delete_student(student_id: str):
    """Soft delete a student (mark as inactive)"""
    try:
        from bson import ObjectId
        
        collection = await get_students_collection()
        result = await collection.update_one(
            {"_id": ObjectId(student_id)},
            {
                "$set": {
                    "is_active": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {
            "success": True,
            "message": "Student record deactivated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete student: {str(e)}")

@router.put("/student/{student_id}/reactivate")
async def reactivate_student(student_id: str):
    """Reactivate a deactivated student"""
    try:
        from bson import ObjectId
        
        collection = await get_students_collection()
        result = await collection.update_one(
            {"_id": ObjectId(student_id)},
            {
                "$set": {
                    "is_active": True,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {
            "success": True,
            "message": "Student record reactivated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reactivate student: {str(e)}")
