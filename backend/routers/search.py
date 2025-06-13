from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime
import logging

from models.student import SearchResult, StudentResponse, SearchLog
from services.face_engine import face_engine
from services.matcher import face_matcher
from db.mongodb import get_students_collection, get_logs_collection

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/search_face", response_model=dict)
async def search_face(
    photo: UploadFile = File(..., description="Photo of the child to identify"),
    request: Request = None
):
    """Search for a student by face photo"""
    
    try:
        # Validate file type
        allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        if photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type {photo.content_type}. Allowed: JPEG, PNG, WebP"
            )
        
        # Read photo data
        photo_bytes = await photo.read()
        
        # Validate face quality
        is_valid, message = face_engine.validate_face_quality(photo_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Photo quality issue: {message}")
        
        # Extract face embedding
        query_embedding = face_engine.get_face_embedding(photo_bytes)
        if not query_embedding:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract face embedding from photo. Please ensure face is clearly visible."
            )
        
        # Get all students from database
        collection = await get_students_collection()
        cursor = collection.find({"is_active": True})
        students_data = await cursor.to_list(length=None)
        
        if not students_data:
            return {
                "success": True,
                "match_found": False,
                "message": "No students registered in the system"
            }
        
        # Search for matches
        match_result = face_matcher.match_against_students(query_embedding, students_data)
        
        # Log the search
        log_entry = {
            "search_timestamp": datetime.utcnow(),
            "matched_student_id": None,
            "confidence": None,
            "ip_address": request.client.host if request else None,
            "user_agent": request.headers.get("user-agent") if request else None
        }
        
        if match_result:
            student_data, confidence = match_result
            
            # Update log entry
            log_entry["matched_student_id"] = str(student_data["_id"])
            log_entry["confidence"] = confidence
            
            # Prepare student response (exclude embeddings and internal fields)
            student_response = {
                "id": str(student_data["_id"]),
                "name": student_data["name"],
                "age": student_data.get("age"),
                "gender": student_data.get("gender"),
                "school": student_data.get("school"),
                "grade": student_data.get("grade"),
                "emergency_contact": student_data["emergency_contact"],
                "medical_info": student_data.get("medical_info"),
                "notes": student_data.get("notes"),
                "created_at": student_data["created_at"],
                "is_active": student_data["is_active"]
            }
            
            # Save search log
            logs_collection = await get_logs_collection()
            await logs_collection.insert_one(log_entry)
            
            return {
                "success": True,
                "match_found": True,
                "student": student_response,
                "confidence": round(confidence, 3),
                "match_quality": get_match_quality(confidence),
                "message": f"Student identified with {round(confidence * 100, 1)}% confidence"
            }
        else:
            # Save search log for no match
            logs_collection = await get_logs_collection()
            await logs_collection.insert_one(log_entry)
            
            return {
                "success": True,
                "match_found": False,
                "message": "No matching student found in the database",
                "suggestion": "Please check if the student is registered or try with a clearer photo"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search_face/batch", response_model=dict)
async def batch_search_face(
    photo: UploadFile = File(..., description="Photo of the child to identify"),
    top_k: int = 3,
    request: Request = None
):
    """Search for top K matching students by face photo"""
    
    try:
        # Validate parameters
        if top_k < 1 or top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
        
        # Validate file type
        allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        if photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type {photo.content_type}. Allowed: JPEG, PNG, WebP"
            )
        
        # Read photo data
        photo_bytes = await photo.read()
        
        # Extract face embedding
        query_embedding = face_engine.get_face_embedding(photo_bytes)
        if not query_embedding:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract face embedding from photo"
            )
        
        # Get all students from database
        collection = await get_students_collection()
        cursor = collection.find({"is_active": True})
        students_data = await cursor.to_list(length=None)
        
        if not students_data:
            return {
                "success": True,
                "matches": [],
                "message": "No students registered in the system"
            }
        
        # Get top matches
        matches = face_matcher.batch_match(query_embedding, students_data, top_k)
        
        # Format response
        formatted_matches = []
        for student_data, confidence in matches:
            student_response = {
                "id": str(student_data["_id"]),
                "name": student_data["name"],
                "age": student_data.get("age"),
                "school": student_data.get("school"),
                "emergency_contact": student_data["emergency_contact"],
                "confidence": round(confidence, 3),
                "match_quality": get_match_quality(confidence)
            }
            formatted_matches.append(student_response)
        
        return {
            "success": True,
            "matches": formatted_matches,
            "total_matches": len(formatted_matches),
            "message": f"Found {len(formatted_matches)} potential matches"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@router.get("/search_logs")
async def get_search_logs(limit: int = 50, offset: int = 0):
    """Get recent search logs"""
    try:
        logs_collection = await get_logs_collection()
        
        # Get logs with pagination, sorted by timestamp (newest first)
        cursor = logs_collection.find().sort("search_timestamp", -1).skip(offset).limit(limit)
        logs = await cursor.to_list(length=None)
        
        # Format logs for response
        formatted_logs = []
        for log in logs:
            formatted_log = {
                "id": str(log["_id"]),
                "timestamp": log["search_timestamp"],
                "matched_student_id": log.get("matched_student_id"),
                "confidence": log.get("confidence"),
                "ip_address": log.get("ip_address"),
                "has_match": log.get("matched_student_id") is not None
            }
            formatted_logs.append(formatted_log)
        
        # Get total count
        total_count = await logs_collection.count_documents({})
        
        return {
            "success": True,
            "logs": formatted_logs,
            "total_count": total_count,
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")

@router.get("/search_stats")
async def get_search_stats():
    """Get search statistics"""
    try:
        logs_collection = await get_logs_collection()
        
        # Total searches
        total_searches = await logs_collection.count_documents({})
        
        # Successful matches
        successful_matches = await logs_collection.count_documents({"matched_student_id": {"$ne": None}})
        
        # Recent searches (last 24 hours)
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_searches = await logs_collection.count_documents({"search_timestamp": {"$gte": yesterday}})
        
        # Success rate
        success_rate = (successful_matches / total_searches * 100) if total_searches > 0 else 0
        
        return {
            "success": True,
            "stats": {
                "total_searches": total_searches,
                "successful_matches": successful_matches,
                "success_rate": round(success_rate, 2),
                "recent_searches_24h": recent_searches
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

def get_match_quality(confidence: float) -> str:
    """Determine match quality based on confidence score"""
    if confidence >= 0.85:
        return "Excellent"
    elif confidence >= 0.75:
        return "Very Good"
    elif confidence >= 0.65:
        return "Good"
    elif confidence >= 0.55:
        return "Fair"
    else:
        return "Poor"
