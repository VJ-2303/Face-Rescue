from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema

class EmergencyContact(BaseModel):
    guardian_name: str = Field(..., description="Guardian's full name")
    phone: str = Field(..., description="Primary contact number")
    alternate_phone: Optional[str] = Field(None, description="Alternate contact number")
    relationship: str = Field(..., description="Relationship to child (parent, guardian, etc.)")
    address: Optional[str] = Field(None, description="Home address")

class MedicalInfo(BaseModel):
    conditions: Optional[List[str]] = Field([], description="Medical conditions")
    medications: Optional[List[str]] = Field([], description="Current medications")
    allergies: Optional[List[str]] = Field([], description="Known allergies")
    special_needs: Optional[str] = Field(None, description="Special care instructions")

class StudentCreate(BaseModel):
    name: str = Field(..., description="Student's full name")
    age: Optional[int] = Field(None, description="Student's age")
    gender: Optional[str] = Field(None, description="Student's gender")
    school: Optional[str] = Field(None, description="School name")
    grade: Optional[str] = Field(None, description="Grade/Class")
    emergency_contact: EmergencyContact
    medical_info: Optional[MedicalInfo] = Field(None, description="Medical information")
    notes: Optional[str] = Field(None, description="Additional notes")

class Student(StudentCreate):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    embeddings: List[List[float]] = Field(..., description="Face embeddings")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, description="Whether the record is active")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class StudentResponse(BaseModel):
    id: str
    name: str
    age: Optional[int]
    gender: Optional[str]
    school: Optional[str]
    grade: Optional[str]
    emergency_contact: EmergencyContact
    medical_info: Optional[MedicalInfo]
    notes: Optional[str]
    created_at: datetime
    is_active: bool

class SearchResult(BaseModel):
    student: StudentResponse
    confidence: float = Field(..., description="Match confidence score (0-1)")
    match_type: str = Field(..., description="Type of match found")

class SearchLog(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    search_timestamp: datetime = Field(default_factory=datetime.utcnow)
    matched_student_id: Optional[str] = Field(None, description="ID of matched student")
    confidence: Optional[float] = Field(None, description="Match confidence")
    ip_address: Optional[str] = Field(None, description="Searcher's IP")
    user_agent: Optional[str] = Field(None, description="Browser/device info")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
