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
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class EmergencyContact(BaseModel):
    guardian_name: str = Field(..., description="Guardian's full name")
    relationship: str = Field(..., description="Relationship to child")
    phone: str = Field(..., description="Primary phone number")
    alternate_phone: Optional[str] = Field(None, description="Alternate phone number")
    address: str = Field(..., description="Home address")
    email: Optional[str] = Field(None, description="Email address")

class StudentBase(BaseModel):
    name: str = Field(..., description="Student's full name")
    age: Optional[int] = Field(None, description="Student's age")
    gender: Optional[str] = Field(None, description="Student's gender")
    school: Optional[str] = Field(None, description="School name")
    class_grade: Optional[str] = Field(None, description="Class/Grade")
    special_needs: Optional[str] = Field(None, description="Any special needs or conditions")
    emergency_contact: EmergencyContact
    additional_info: Optional[str] = Field(None, description="Additional information")

class StudentCreate(StudentBase):
    pass

class StudentInDB(StudentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    face_embeddings: List[List[float]] = Field(default_factory=list, description="Face embedding vectors")
    image_count: int = Field(default=0, description="Number of face images stored")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class StudentResponse(StudentBase):
    id: str = Field(alias="_id")
    image_count: int
    created_at: datetime
    is_active: bool

    class Config:
        allow_population_by_field_name = True

class FaceSearchResult(BaseModel):
    student: StudentResponse
    confidence: float = Field(..., description="Confidence score (0-1)")
    match_found: bool = Field(..., description="Whether a match was found above threshold")

class FaceSearchResponse(BaseModel):
    success: bool
    message: str
    result: Optional[FaceSearchResult] = None
    processing_time: float = Field(..., description="Processing time in seconds")
