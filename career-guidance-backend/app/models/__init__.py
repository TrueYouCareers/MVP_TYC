# File: /career-guidance-backend/career-guidance-backend/app/models/__init__.py

from app.db.session import Base
# Import all your models here to ensure they are registered with Base
from .user import User
from .profile import Profile
