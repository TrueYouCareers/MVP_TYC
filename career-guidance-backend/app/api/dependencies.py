import os
from app.llm.rag_system import RAGSystem  # Ensure RAGSystem is importable

# Global RAG system instance
_rag_system_instance = None


def get_rag_system_for_profile() -> RAGSystem:
    """
    Initializes and returns a singleton instance of the RAGSystem.
    Requires GROQ_API_KEY environment variable to be set.
    """
    global _rag_system_instance
    if _rag_system_instance is None:
        if not os.getenv("GROQ_API_KEY"):
            # This should ideally be checked at app startup
            # For now, raising an error here is fine for the dependency
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        # Assuming RAGSystem is initialized like this. Adjust if necessary.
        _rag_system_instance = RAGSystem(knowledge_dir="knowledge_base")
    return _rag_system_instance

# Add other common dependencies here if needed, e.g., get_current_active_user
# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt
# from app.core.config import settings # Assuming you have a config file for SECRET_KEY etc.
# from app.models.user import User
# from app.db.session import SessionLocal # If using SQL DB for users

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)): # get_db would be another dependency
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#         token_data = TokenData(username=username) # Define TokenData schema
#     except JWTError:
#         raise credentials_exception
#     user = db.query(User).filter(User.email == token_data.username).first() # Adjust for your user model and query
#     if user is None:
#         raise credentials_exception
#     return user

# async def get_current_active_user(current_user: User = Depends(get_current_user)):
#     if not current_user.is_active: # Assuming an is_active field on your User model
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user
