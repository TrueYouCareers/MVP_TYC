from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Changed: Import from connections
from app.db.connections import get_db_connections as get_db
from app.schemas.chat import ChatMessageInput, ChatSession
from app.schemas.questionnaire import ProfileUpdateData
from app.services.chat_service import ChatService
from app.services.profile_service import ProfileService
from app.services.profile_analysis_service import ProfileAnalysisService
from app.llm.rag_system import RAGSystem
import os

from logging import getLogger
logger = getLogger(__name__)

router = APIRouter()

# --- Dependency for RAGSystem ---
_rag_system_instance: Optional[RAGSystem] = None


def get_rag_system_dependency():
    global _rag_system_instance
    if _rag_system_instance is None:
        if not os.getenv("OPEN_AI_API_KEY"):
            raise RuntimeError("OPEN_AI_API_KEY not set in environment.")
        _rag_system_instance = RAGSystem(knowledge_dir="knowledge_base")
    return _rag_system_instance

# Need to add ChatResponse model since it's missing


class ChatResponse(BaseModel):
    session_id: str
    response: str
    source_documents: List[Dict[str, Any]] = []


@router.post("/chat", response_model=ChatResponse)
async def chat_with_guidance_system(
    chat_input: ChatMessageInput,
    db_clients=Depends(get_db),
    rag_system: RAGSystem = Depends(get_rag_system_dependency)
):
    logger.info(f"=== CHAT ENDPOINT DEBUG ===")
    logger.info(f"Request received for user_id: {chat_input.user_id}")
    logger.info(f"Question: {chat_input.question[:100]}...")
    logger.info(f"Session ID: {chat_input.session_id}")

    try:
        # Extract MongoDB client correctly
        mysql_client, mongodb_client = db_clients

        # Step 1: Get user profile with detailed logging
        logger.info("🔍 Step 1: Retrieving user profile...")

        profile_service = ProfileService(mongo_db=mongodb_client)
        user_questionnaire_data = profile_service.get_user_questionnaire(
            chat_input.user_id)

        if user_questionnaire_data:
            logger.info("✅ User questionnaire data found!")
            logger.info(
                f"Education level: {user_questionnaire_data.get('education_level')}")
            logger.info(
                f"Has LLM profile: {'llm_profile' in user_questionnaire_data}")
            logger.info(
                f"LLM profile keys: {list(user_questionnaire_data.get('llm_profile', {}).keys())}")
        else:
            logger.warning("❌ No user questionnaire data found!")

        # Step 2: Extract user profile for RAG system
        logger.info("🔍 Step 2: Extracting profile for RAG system...")

        if user_questionnaire_data and user_questionnaire_data.get("llm_profile"):
            user_profile = user_questionnaire_data["llm_profile"].copy()
            user_profile["education_level"] = user_questionnaire_data.get(
                "education_level", "unknown")

            logger.info("✅ Profile extracted successfully!")
            logger.info(f"Profile contains: {list(user_profile.keys())}")
        else:
            logger.warning(
                "❌ Creating empty profile - this will cause context issues!")
            user_profile = {
                "education_level": "unknown",
                "profile_summary": "",
                "identified_keywords": [],
                "primary_orientation": ""
            }

        # Step 3: Chat session handling - FIXED: Pass rag_system to ChatService
        logger.info("🔍 Step 3: Handling chat session...")

        chat_service = ChatService(
            mongo_db=mongodb_client, rag_system=rag_system)

        if chat_input.session_id:
            logger.info(f"Using existing session: {chat_input.session_id}")
            session_id = chat_input.session_id
        else:
            logger.info("Creating new chat session...")
            session_id = await chat_service.create_chat_session(chat_input.user_id)
            logger.info(f"New session created: {session_id}")

        # Step 4: RAG query with enhanced logging
        logger.info("🔍 Step 4: Querying RAG system...")
        logger.info(f"Profile being sent to RAG: {user_profile}")

        # Use the injected rag_system instead of creating new instance
        rag_response = rag_system.query(
            question=chat_input.question,
            user_profile=user_profile,
            stream=False
        )

        logger.info("✅ RAG response received!")
        logger.info(f"Response length: {len(rag_response.get('answer', ''))}")

        # Step 5: Save chat message
        logger.info("🔍 Step 5: Saving chat message...")

        await chat_service.add_message_to_session(
            session_id=session_id,
            message_type="user",
            content=chat_input.question
        )

        await chat_service.add_message_to_session(
            session_id=session_id,
            message_type="assistant",
            content=rag_response["answer"]
        )

        logger.info("✅ Chat message saved successfully!")
        logger.info("=== CHAT ENDPOINT COMPLETE ===")

        return ChatResponse(
            session_id=session_id,
            response=rag_response["answer"],
            source_documents=[
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in rag_response.get("source_documents", [])
            ]
        )

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@router.get("/chat/history/{user_id}/{session_id}", response_model=ChatSession)
async def get_chat_session_history(
    user_id: str,
    session_id: str,
    db_clients=Depends(get_db),
    rag_system: RAGSystem = Depends(get_rag_system_dependency)
):
    try:
        mysql_client, mongodb_client = db_clients
        chat_service = ChatService(
            mongo_db=mongodb_client, rag_system=rag_system)
        session = await chat_service.get_chat_history(user_id, session_id)
        if not session:
            raise HTTPException(
                status_code=404, detail="Chat session not found.")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chat history: {str(e)}")


@router.get("/chat/sessions/{user_id}", response_model=List[ChatSession])
async def get_all_user_chat_sessions(
    user_id: str,
    db_clients=Depends(get_db),
    rag_system: RAGSystem = Depends(get_rag_system_dependency)
):
    try:
        mysql_client, mongodb_client = db_clients
        chat_service = ChatService(
            mongo_db=mongodb_client, rag_system=rag_system)
        sessions = await chat_service.get_user_sessions(user_id)
        return sessions
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving user sessions: {str(e)}")


class IntelligentUpdateRequest(BaseModel):
    user_id: str
    new_information: str


@router.post("/profile/intelligent-update", response_model=ProfileUpdateData)
async def intelligent_profile_update_endpoint(
    request_data: IntelligentUpdateRequest,
    db_clients=Depends(get_db),
    rag_system: RAGSystem = Depends(get_rag_system_dependency)
):
    try:
        mysql_client, mongodb_client = db_clients

        analysis_service = ProfileAnalysisService(
            mongo_db=mongodb_client, rag_system=rag_system)
        updated_profile_data_obj = await analysis_service.intelligently_update_profile(
            request_data.user_id, request_data.new_information
        )

        profile_service = ProfileService(mongo_db=mongodb_client)
        success = await profile_service.update_user_profile(request_data.user_id, updated_profile_data_obj)

        if success:
            return updated_profile_data_obj
        else:
            raise HTTPException(
                status_code=404,
                detail="User profile not found for update, or no changes were applied."
            )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during intelligent profile update: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during intelligent profile update: {str(e)}"
        )
