from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional
import json
import logging
from datetime import datetime

from db.mongodb import get_database
from models.student import StudentCreate, StudentInDB, StudentResponse, EmergencyContact
from services.simple_face_engine import simple_face_processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/students", tags=["students"])

@router.post("/register", response_model=dict)
async def register_student(
    name: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    school: Optional[str] = Form(None),
    class_grade: Optional[str] = Form(None),
    special_needs: Optional[str] = Form(None),
    guardian_name: str = Form(...),
    relationship: str = Form(...),
    phone: str = Form(...),
    alternate_phone: Optional[str] = Form(None),
    address: str = Form(...),
    email: Optional[str] = Form(None),
    additional_info: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
    db=Depends(get_database)
):
    """
    Register a new student with face images
    """
    try:
        # Validate images
        if len(images) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 face images are required for accurate identification"
            )
        
        if len(images) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 10 images allowed"
            )
        
        # Validate image formats
        allowed_formats = {"image/jpeg", "image/jpg", "image/png"}
        for img in images:
            if img.content_type not in allowed_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image format: {img.content_type}. Only JPEG and PNG are allowed."
                )
        
        logger.info(f"Starting registration for student: {name}")
        
        # Read image data
        image_data_list = []
        for img in images:
            content = await img.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty image file detected")
            image_data_list.append(content)
        
        # Extract face embeddings
        logger.info("Extracting face embeddings...")
        embeddings = simple_face_processor.extract_multiple_embeddings(image_data_list)
        
        if len(embeddings) == 0:
            raise HTTPException(
                status_code=400,
                detail="No faces detected in any of the uploaded images. Please ensure clear face photos."
            )
        
        if len(embeddings) < len(images) * 0.6:  # At least 60% success rate
            raise HTTPException(
                status_code=400,
                detail="Too few faces detected. Please upload clearer face images."
            )
        
        # Create emergency contact
        emergency_contact = EmergencyContact(
            guardian_name=guardian_name,
            relationship=relationship,
            phone=phone,
            alternate_phone=alternate_phone,
            address=address,
            email=email
        )
        
        # Create student document
        student_doc = StudentInDB(
            name=name,
            age=age,
            gender=gender,
            school=school,
            class_grade=class_grade,
            special_needs=special_needs,
            emergency_contact=emergency_contact,
            additional_info=additional_info,
            face_embeddings=embeddings,
            image_count=len(embeddings),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert into database
        students_collection = db.students
        result = await students_collection.insert_one(student_doc.dict(by_alias=True))
        
        logger.info(f"Student registered successfully with ID: {result.inserted_id}")
        
        return {
            "success": True,
            "message": f"Student '{name}' registered successfully",
            "student_id": str(result.inserted_id),
            "embeddings_count": len(embeddings),
            "images_processed": len(images)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering student: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/list", response_model=List[StudentResponse])
async def list_students(
    skip: int = 0,
    limit: int = 50,
    db=Depends(get_database)
):
    """
    List all registered students
    """
    try:
        students_collection = db.students
        
        cursor = students_collection.find(
            {"is_active": True},
            {"face_embeddings": 0}  # Exclude embeddings for performance
        ).skip(skip).limit(limit)
        students = []
        async for student_doc in cursor:
            # Convert ObjectId to string and rename _id to id
            if "_id" in student_doc:
                student_doc["id"] = str(student_doc["_id"])
                del student_doc["_id"]
            students.append(StudentResponse(**student_doc))
        
        return students
        
    except Exception as e:
        logger.error(f"Error listing students: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve students")

@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str, db=Depends(get_database)):
    """
    Get student details by ID
    """
    try:
        from bson import ObjectId
        
        students_collection = db.students
        student_doc = await students_collection.find_one(
            {"_id": ObjectId(student_id)},
            {"face_embeddings": 0}  # Exclude embeddings
        )
        
        if not student_doc:
            raise HTTPException(status_code=404, detail="Student not found")
          # Convert ObjectId to string and rename _id to id
        if "_id" in student_doc:
            student_doc["id"] = str(student_doc["_id"])
            del student_doc["_id"]
        return StudentResponse(**student_doc)
        
    except Exception as e:
        logger.error(f"Error getting student: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve student")

@router.delete("/{student_id}")
async def delete_student(student_id: str, db=Depends(get_database)):
    """
    Soft delete a student (mark as inactive)
    """
    try:
        from bson import ObjectId
        
        students_collection = db.students
        result = await students_collection.update_one(
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
        
        return {"success": True, "message": "Student deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete student")
