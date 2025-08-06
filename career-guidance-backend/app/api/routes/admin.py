from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.db.connections import get_db_connections
from app.services.admin_service import AdminService
from app.schemas.user import User as UserSchema, UserWithResponses

router = APIRouter()

@router.get("/users", response_model=List[UserSchema])
def get_all_users(db: Session = Depends(get_db),db_clients=Depends(get_db_connections)):
    _, mongo = db_clients
    admin_service = AdminService(db=db, mongo_db=mongo)
    try:
        return admin_service.get_all_users()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")

@router.get("/users/{user_id}", response_model=UserWithResponses)
def get_user_details(user_id: str, db : Session =Depends(get_db), db_clients=Depends(get_db_connections)):
    _, mongo = db_clients
    admin_service = AdminService(db=db, mongo_db=mongo)
    try: 
        return admin_service.get_user_details(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")