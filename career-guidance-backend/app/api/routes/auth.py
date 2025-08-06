from app.core.security import ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session  # Import Session
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
# Changed: Import from connections
from app.db.connections import get_db_connections 
from app.schemas.user import UserCreate, User as UserResponseSchema  # Renamed for clarity
# Assuming you have a Token schema for JWT response
from app.schemas.auth import Token
from app.services.auth_service import AuthService
from app.services.profile_service import ProfileService
from app.db.session import get_db  # Use SQLAlchemy session getter
from app.core.security import create_access_token
from datetime import timedelta


router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_mongodb_client():
    _, mongodb_client = get_db_connections()
    return mongodb_client

# Pydantic model for login request, if not already in schemas.auth
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup", response_model=UserResponseSchema)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    auth_service = AuthService(db=db)
    try:
        created_user = auth_service.register_user(user_create=user)
        return created_user
    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the exception e
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred during registration. {e}")


@router.post("/login", response_model=Token)
async def login(form_data: LoginRequest, db: Session = Depends(get_db)):  
    auth_service = AuthService(db=db)
    profile_service = ProfileService(mongo_db=get_mongodb_client())
    try:
        user = auth_service.login_user(
            email=form_data.email, password=form_data.password)
        if not user:  
            raise HTTPException(status_code=401, detail="Invalid credentials")
        questionnaire_data = await profile_service.get_questionnaire_by_user_id(str(user.id))

        print({
        "user_id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "contact": user.contact,
        "llm_profile": questionnaire_data.llm_profile
        })

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "user_id": str(user.id)}, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token, 
            "token_type": "bearer", "user_id": str(user.id), 
            "education_level": user.education_level, 
            "contact": user.contact,
            "full_name": user.full_name,
            "email": user.email,
            "llm_profile": questionnaire_data.llm_profile
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during login.")


