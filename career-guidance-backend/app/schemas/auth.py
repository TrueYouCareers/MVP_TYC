from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    education_level: str
    contact: str 
    full_name: str 
    email: str
    llm_profile: Dict[str, Any]


class TokenData(BaseModel):
    email: Optional[EmailStr] = None
    user_id: Optional[str] = None
