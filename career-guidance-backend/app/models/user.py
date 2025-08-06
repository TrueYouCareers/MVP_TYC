from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base
import uuid  # For UUID primary key
from sqlalchemy.dialects.mysql import CHAR  # For UUID in MySQL


class User(Base):
    __tablename__ = "users"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    contact = Column(String(50))
    education_level = Column(String(100))

    # Relationship to Profile (if one-to-one or one-to-many)
    # Example for one-to-one:
    profile = relationship("Profile", back_populates="user",
                           uselist=False, cascade="all, delete-orphan")
    # For one-to-many (if a user can have multiple profiles, though less likely):
    # profiles = relationship("Profile", back_populates="user", cascade="all, delete-orphan")

    # Add other fields as per your users table structure
    # e.g. is_active = Column(Boolean, default=True)
