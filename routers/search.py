from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import logging
import time
from typing import Optional

from db.mongodb import get_database
from models.student import FaceSearchResponse, FaceSearchResult, StudentResponse
from services.simple_accurate_face_engine import simple_accurate_face_processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])

@router.post("/face", response_model=FaceSearchResponse)
async def search_face(
    image: UploadFile = File(...),
    threshold: Optional[float] = 0.7,
    db=Depends(get_database)
):
    """
    Search for a student by face image
    """
    start_time = time.time()
    
    try:
        # Validate image
        if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Only JPEG and PNG are allowed."
            )
        
        # Read image data
        image_data = await image.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        logger.info("Processing search image...")
          # Extract face embedding from search image
        search_embedding = simple_accurate_face_processor.extract_face_embedding(image_data)
        
        if not search_embedding:
            processing_time = time.time() - start_time
            return FaceSearchResponse(
                success=False,
                message="No face detected in the uploaded image. Please ensure the image contains a clear face.",
                result=None,
                processing_time=processing_time
            )
        
        logger.info("Face embedding extracted successfully")
        
        # Get all active students from database
        students_collection = db.students
        cursor = students_collection.find({"is_active": True})
        
        best_match = None
        best_confidence = 0.0
        students_checked = 0
        
        async for student_doc in cursor:
            students_checked += 1
            stored_embeddings = student_doc.get("face_embeddings", [])
            
            if not stored_embeddings:
                continue
              # Compare with this student's embeddings
            for stored_embedding in stored_embeddings:
                similarity = simple_accurate_face_processor.compare_faces(search_embedding, stored_embedding)
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = student_doc
        
        processing_time = time.time() - start_time
        logger.info(f"Search completed. Checked {students_checked} students in {processing_time:.2f}s")        # Check if we found a valid match
        if best_match and best_confidence >= simple_accurate_face_processor.confidence_threshold:            # Prepare student response (exclude embeddings and convert ObjectId)
            best_match.pop("face_embeddings", None)
            
            # Convert ObjectId to string and rename _id to id
            if "_id" in best_match:
                best_match["id"] = str(best_match["_id"])
                del best_match["_id"]
            
            student_response = StudentResponse(**best_match)
            
            result = FaceSearchResult(
                student=student_response,
                confidence=best_confidence,
                match_found=True
            )
            
            logger.info(f"Match found: {best_match['name']} with confidence {best_confidence:.3f}")
            
            return FaceSearchResponse(
                success=True,
                message=f"Student identified: {best_match['name']}",
                result=result,
                processing_time=processing_time
            )
        
        else:
            # No match found above threshold
            if best_match:
                logger.info(f"Best match confidence {best_confidence:.3f} below threshold {threshold}")
                message = f"No confident match found. Best similarity: {best_confidence:.3f}"
            else:
                message = "No students found in database"
            
            return FaceSearchResponse(
                success=True,
                message=message,
                result=None,
                processing_time=processing_time
            )
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in face search: {e}")
        return FaceSearchResponse(
            success=False,
            message=f"Search failed: {str(e)}",
            result=None,
            processing_time=processing_time
        )

@router.get("/stats")
async def get_search_stats(db=Depends(get_database)):
    """
    Get search statistics
    """
    try:
        students_collection = db.students
        
        total_students = await students_collection.count_documents({"is_active": True})
        total_embeddings = 0
        
        cursor = students_collection.find({"is_active": True}, {"face_embeddings": 1})
        async for doc in cursor:
            embeddings = doc.get("face_embeddings", [])
            total_embeddings += len(embeddings)
        
        return {
            "total_students": total_students,
            "total_face_embeddings": total_embeddings,
            "average_images_per_student": total_embeddings / max(total_students, 1),            "confidence_threshold": simple_accurate_face_processor.confidence_threshold,
            "model_info": {
                "face_model": "Simple OpenCV Face Detection",
                "detector": "opencv"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")
