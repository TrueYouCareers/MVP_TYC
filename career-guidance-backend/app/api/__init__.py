# Contents of /career-guidance-backend/career-guidance-backend/app/api/__init__.py

from fastapi import APIRouter

router = APIRouter()

from .routes import auth

router.include_router(auth.router, prefix="/auth", tags=["auth"])
# router.include_router(guidance.router, prefix="/guidance", tags=["guidance"])
# router.include_router(users.router, prefix="/users", tags=["users"])