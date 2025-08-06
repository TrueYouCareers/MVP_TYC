from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.user import User  # Changed import
from app.schemas.user import UserCreate, User as UserSchema  # Changed import and alias
from app.core.security import get_password_hash, verify_password  # Changed import
# from utils.token_counter import count_tokens # This utility is not defined, commenting out for now


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def register_user(self, user_create: UserCreate) -> UserSchema:
        existing_user = self.db.query(User).filter(
            User.email == user_create.email).first()
        if existing_user:
            raise HTTPException(
                status_code=400, detail="Email already registered")

        hashed_password = get_password_hash(user_create.password)
        # Ensure username is derived if not provided, or make it mandatory in UserCreate
        username = user_create.username or user_create.email.split("@")[0]

        new_user = User(
            email=user_create.email,
            hashed_password=hashed_password,
            username=username,
            full_name=user_create.full_name,
            contact=user_create.contact,  # Will be None if not provided in UserCreate
            education_level=user_create.education_level,  # Will be None if not provided
        )
        self.db.add(new_user)
        self.db.commit()
        self.db.refresh(new_user)
        return UserSchema.from_orm(new_user)

    def login_user(self, email: str, password: str) -> UserSchema:
        user = self.db.query(User).filter(User.email == email).first()
        # Ensure field name matches model
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return UserSchema.from_orm(user)

    # def count_user_tokens(self, user_id: str) -> int: # user_id is UUID string
    #     user = self.db.query(User).filter(User.id == user_id).first()
    #     if not user:
    #         raise HTTPException(status_code=404, detail="User not found")
    #     # return count_tokens(user.email)  # Example of counting tokens based on user email
    #     return 0 # Placeholder as count_tokens is not defined
