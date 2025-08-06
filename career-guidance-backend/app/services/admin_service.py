from pymongo.database import Database
from pymongo.errors import PyMongoError
from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models.user import User as UserModel
from app.schemas.user import UserWithResponses
from app.schemas.user import User as UserSchema
from uuid import UUID 


class AdminService:
    def __init__(self, db: Session,  mongo_db: Database):
        self.db = db
        self.mongo_db = mongo_db
        self.collection = self.mongo_db["user_questionnaires"]

    def get_all_users(self) -> List[UserSchema]:
        users = self.db.query(UserModel).all()
        if not users:
            raise HTTPException(status_code=404, detail="No users found")
        return [UserSchema.from_orm(user) for user in users]
    
    def get_user_details(self, user_id: UUID) -> UserWithResponses:
        user = self.db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        mongo_doc = self.collection.find_one({"user_id": str(user_id)}, sort=[("timestamp", -1)])
        raw_responses = mongo_doc.get("raw_responses", {}) if mongo_doc else {}
        if not raw_responses:
            raise HTTPException(status_code=405, detail="Raw responses not found")

        return UserWithResponses(
            user=UserSchema.from_orm(user), 
            raw_responses=raw_responses
        )
        
