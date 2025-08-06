from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base
from sqlalchemy.dialects.mysql import CHAR  # For UUID foreign key


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    education_level = Column(String(255), nullable=False)
    interest_profile = Column(String(1000))  # Assuming text field
    dominant_subjects = Column(String(1000))  # Assuming text field

    user_id = Column(CHAR(36), ForeignKey("users.id"), unique=True,
                     nullable=False)  # unique=True for one-to-one
    user = relationship("User", back_populates="profile")
