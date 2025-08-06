from pydantic import BaseModel
from typing import Optional


class ProfileBase(BaseModel):
    education_level: str
    interest_profile: str
    dominant_subjects: str


class ProfileCreate(ProfileBase):
    pass


class ProfileUpdate(ProfileBase):
    education_level: Optional[str] = None
    interest_profile: Optional[str] = None
    dominant_subjects: Optional[str] = None


class Profile(ProfileBase):
    id: int

    class Config:
        from_attributes = True
