from pymongo.database import Database
from app.schemas.chat import ChatMessageInput, ChatSession, ChatMessage
from app.llm.rag_system import RAGSystem
from app.llm.prompts import Prompts
from bson import ObjectId
from datetime import datetime
import uuid
import json
from typing import Dict, Any, List, Optional


class ChatService:
    def __init__(self, mongo_db: Database, rag_system: Optional[RAGSystem] = None):
        self.mongo_db = mongo_db
        self.rag_system = rag_system
        self.collection = self.mongo_db["chat_sessions"]

    async def create_chat_session(self, user_id: str) -> str:
        """Create a new chat session and return the session ID"""
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": [],
            "summary": None
        }
        insert_result = self.collection.insert_one(session_data)
        return str(insert_result.inserted_id)

    async def add_message_to_session(self, session_id: str, message_type: str, content: str):
        """Add a message to an existing chat session"""
        message = {
            "role": message_type,
            "content": content,
            "timestamp": datetime.utcnow()
        }

        self.collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def handle_chat_message(self, chat_input: ChatMessageInput, user_llm_profile: Dict[str, Any]) -> ChatSession:
        session_id_to_find = chat_input.session_id
        session_doc = None

        if session_id_to_find:
            try:
                session_doc = self.collection.find_one(
                    {"_id": ObjectId(session_id_to_find), "user_id": chat_input.user_id})
                if not session_doc:
                    session_id_to_find = None
            except Exception:
                session_id_to_find = None

        if not session_id_to_find:
            session_data_for_insert = {
                "user_id": chat_input.user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "messages": [],
                "summary": None
            }
            insert_result = self.collection.insert_one(session_data_for_insert)
            session_id_to_find = str(insert_result.inserted_id)
            session_doc = self.collection.find_one(
                {"_id": ObjectId(session_id_to_find)})

        session = ChatSession(**session_doc, id=str(session_doc["_id"]))

        # Extract chat history before adding the new message
        chat_history = [msg.model_dump() for msg in session.messages]

        # Use the injected RAG system
        if self.rag_system:
            llm_response_data = self.rag_system.query(
                question=chat_input.question, user_profile=user_llm_profile, chat_history=chat_history)
            ai_content = llm_response_data.get(
                "answer", "Sorry, I encountered an issue and cannot respond at the moment.")
        else:
            ai_content = "RAG system not available. Please contact support."

        user_message = ChatMessage(
            role="user", content=chat_input.question, timestamp=datetime.utcnow())
        session.messages.append(user_message)

        ai_message = ChatMessage(
            role="ai", content=ai_content, timestamp=datetime.utcnow())
        session.messages.append(ai_message)
        session.updated_at = datetime.utcnow()

        self.collection.update_one(
            {"_id": ObjectId(session.id)},
            {"$set": {
                "messages": [msg.model_dump() for msg in session.messages],
                "updated_at": session.updated_at
            }}
        )
        return session

    async def get_chat_history(self, user_id: str, session_id: str) -> Optional[ChatSession]:
        try:
            session_doc = self.collection.find_one(
                {"_id": ObjectId(session_id), "user_id": user_id})
            if session_doc:
                session_doc["id"] = str(session_doc["_id"])
                return ChatSession(**session_doc)
        except Exception:  # Invalid ObjectId or other error
            return None
        return None

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        sessions_cursor = self.collection.find(
            {"user_id": user_id}).sort("updated_at", -1)
        sessions = []
        for doc in sessions_cursor:
            doc["id"] = str(doc["_id"])
            sessions.append(ChatSession(**doc))
        return sessions

    async def summarize_chat_session(self, session_id: str) -> Optional[str]:
        try:
            session_doc = self.collection.find_one(
                {"_id": ObjectId(session_id)})
        except Exception:
            return None

        if not session_doc:
            return None

        messages = [ChatMessage(**msg)
                    for msg in session_doc.get("messages", [])]
        if not messages:
            return ""

        formatted_chat_history = "\n".join(
            [f"{m.role}: {m.content}" for m in messages])

        prompt_template = Prompts.get_chat_summarization_prompt()
        prompt = prompt_template.format(chat_history=formatted_chat_history)

        try:
            # Assuming rag_system.llm is the ChatGroq instance
            llm_response = self.rag_system.llm.invoke(prompt)
            summary = llm_response.content if hasattr(
                llm_response, 'content') else str(llm_response)
        except Exception as e:
            # Log error e
            summary = f"Could not summarize. Session has {len(messages)} messages. Last: '{messages[-1].content[:50]}...'"

        self.collection.update_one({"_id": ObjectId(session_id)}, {
                                   "$set": {"summary": summary, "updated_at": datetime.utcnow()}})
        return summary
