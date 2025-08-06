from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
# Changed: Import from connections
from app.db.connections import get_db_connections as get_db
from app.schemas.questionnaire import QuestionnaireSubmission, ProfileUpdateData
from app.services.profile_service import ProfileService
from app.schemas.questionnaire import QuestionnaireInDB, QuestionnaireResponse
# Removed: from app.llm.rag_system import RAGSystem
import logging
# Removed: import os
from app.data.questions import QUESTIONNAIRE_DATA
from app.api.dependencies import get_rag_system_for_profile  # Updated import
from app.llm.rag_system import RAGSystem  # Ensure RAGSystem is importable
# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


# Removed: Global RAG system instance and its direct initialization function
# _rag_system_instance = None
# def get_rag_system_for_profile(): ... (this logic is now in dependencies.py)


@router.post("/questionnaire", response_model=QuestionnaireResponse)
def save_user_profile(
    profile_data: QuestionnaireSubmission,
    db_clients=Depends(get_db),
    rag_system: RAGSystem = Depends(
        get_rag_system_for_profile)  # Use common dependency
):
    try:
        _, mongodb_client = db_clients
        profile_service = ProfileService(mongo_db=mongodb_client)

        # RAG system is now injected by FastAPI
        logger.info(f"Generating LLM profile for user {profile_data.user_id}")
        llm_profile = rag_system.generate_llm_student_profile(
            education_level=profile_data.education_level,
            responses=profile_data.raw_responses,
            questions_data=profile_data.questions_data  # This can be None
        )

        # Create complete questionnaire data
        complete_data = QuestionnaireInDB(
            user_id=profile_data.user_id,
            education_level=profile_data.education_level,
            raw_responses=profile_data.raw_responses,
            llm_profile=llm_profile
        )

        # Save to database
        inserted_id = profile_service.save_complete_questionnaire(
            complete_data)
        logger.info(
            f"Successfully saved profile for user {profile_data.user_id} with ID {inserted_id}")

        return QuestionnaireResponse(
            message="Profile and questionnaire responses saved successfully!",
            id=inserted_id,
            llm_profile=llm_profile
        )

    except Exception as e:
        logger.error(
            f"Error in save_user_profile for user {profile_data.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process questionnaire: {str(e)}")


@router.get("/questionnaire/{user_id}")
async def get_user_questionnaire(user_id: str, db_clients=Depends(get_db)):
    try:
        _, mongodb_client = db_clients
        profile_service = ProfileService(mongo_db=mongodb_client)
        questionnaire_data = await profile_service.get_questionnaire_by_user_id(user_id)
        if questionnaire_data:
            return questionnaire_data
        raise HTTPException(
            status_code=404, detail="Questionnaire data not found for this user.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/questionnaire/{user_id}/profile")
async def update_user_llm_profile(user_id: str, profile_data: ProfileUpdateData, db_clients=Depends(get_db)):
    try:
        _, mongodb_client = db_clients
        profile_service = ProfileService(mongo_db=mongodb_client)

        success = await profile_service.update_user_profile(user_id, profile_data)

        if success:
            return {"message": "User LLM profile updated successfully!"}
        else:
            # This could mean the profile wasn't found or no changes were made.
            # Depending on desired behavior, you might want a more specific error if not found.
            raise HTTPException(
                status_code=404, detail="User profile not found or no changes made.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/questions/{education_level}")
def get_questions_for_education_level(education_level: str):
    """Get questionnaire questions for a specific education level"""
    if education_level not in QUESTIONNAIRE_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Questions not found for education level: {education_level}"
        )

    return {
        "education_level": education_level,
        "questions": QUESTIONNAIRE_DATA[education_level]
    }
