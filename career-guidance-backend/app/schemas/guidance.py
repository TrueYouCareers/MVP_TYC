from pydantic import BaseModel
from typing import List, Optional


class GuidanceRequest(BaseModel):
    question: str
    user_profile_id: str


class GuidanceResponse(BaseModel):
    answer: str
    source_documents: List[str]
    token_usage: dict


class UserProfile(BaseModel):
    education_level: str
    interest_profile: str
    dominant_subjects: str
    response_stats: dict


class GuidanceQuery(BaseModel):
    request: GuidanceRequest
    user_profile: UserProfile
    response: Optional[GuidanceResponse] = None
