from fastapi import APIRouter, Depends, HTTPException
from app.db.connections import get_db_connections as get_db
from app.schemas.questionnaire import QuestionnaireSubmission, ProfileUpdateData
from app.services.profile_service import ProfileService
from app.schemas.questionnaire import QuestionnaireInDB, QuestionnaireResponse
from app.llm.rag_system import RAGSystem
import logging
import os

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Global RAG system instance for profile generation
_rag_system_instance = None


def get_rag_system_for_profile():
    global _rag_system_instance
    if _rag_system_instance is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        _rag_system_instance = RAGSystem(knowledge_dir="knowledge_base")
    return _rag_system_instance


@router.post("/questionnaire", response_model=QuestionnaireResponse)
def save_user_profile(profile_data: QuestionnaireSubmission, db_clients=Depends(get_db)):
    try:
        logger.info(
            f"Received questionnaire data: user_id={profile_data.user_id}, education_level={profile_data.education_level}"
        )
        logger.info(
            f"Raw responses keys: {list(profile_data.raw_responses.keys())}"
        )

        _, mongodb_client = db_clients
        profile_service = ProfileService(mongo_db=mongodb_client)

        # Initialize RAG system to generate LLM profile
        rag_system = get_rag_system_for_profile()

        # Generate LLM profile from responses
        logger.info(f"Generating LLM profile for user {profile_data.user_id}")
        if profile_data.education_level == "10th":
            llm_profile = rag_system.generate_llm_student_profile_10(
                education_level=profile_data.education_level,
                responses=profile_data.raw_responses,
                questions_data=profile_data.questions_data  # This can be None
            )
        elif profile_data.education_level == "12th":
            llm_profile = rag_system.generate_llm_student_profile_12(
                    education_level=profile_data.education_level,
                    responses=profile_data.raw_responses,
                    questions_data=profile_data.questions_data  # This can be None
            )
        else: 
            llm_profile = rag_system.generate_llm_student_profile_graduate(
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
        questionnaire_data = await profile_service.get_questionnaire_by_user_id(
            user_id)
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
