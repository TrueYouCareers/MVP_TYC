from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import uuid


class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    contact: Optional[str] = None
    education_level: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    password: Optional[str] = None



class User(UserBase):
    id: uuid.UUID  

    class Config:
        from_attributes = True

class UserWithResponses(BaseModel):
    user: User  
    raw_responses: Dict[str, Any]

class Users(BaseModel):
    users: List[User]