from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessageInput(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    question: str  # Changed from 'message' to 'question' to match frontend
    # Optional: pass current user profile context if needed for stateless call
    # current_llm_profile: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "ai"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseModel):
    id: str  # Could be MongoDB ObjectId as string, or UUID
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = []
    summary: Optional[str] = None  # For context summarization of the session
    # Optional: store the llm_profile snapshot at the start of this session for context
    # session_context_profile: Optional[Dict[str, Any]] = None
