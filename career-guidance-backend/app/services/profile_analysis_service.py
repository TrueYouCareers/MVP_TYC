from pymongo.database import Database
from app.schemas.questionnaire import ProfileUpdateData
from app.llm.rag_system import RAGSystem
from app.llm.prompts import Prompts
from datetime import datetime
import json
from typing import Dict, Any


class ProfileAnalysisService:
    def __init__(self, mongo_db: Database, rag_system: RAGSystem):
        self.mongo_db = mongo_db
        self.rag_system = rag_system  # LLM interface

    async def intelligently_update_profile(self, user_id: str, new_information_text: str) -> ProfileUpdateData:
        collection = self.mongo_db["user_questionnaires"]
        latest_document = collection.find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)]
        )
        if not latest_document or "llm_profile" not in latest_document:
            raise ValueError(
                f"Existing LLM profile not found for user_id: {user_id}")

        current_llm_profile = latest_document["llm_profile"]
        if not isinstance(current_llm_profile, dict):
            raise ValueError(
                f"LLM profile for user_id: {user_id} is not in the expected format.")

        try:
            original_profile_json_str = json.dumps(current_llm_profile)
        except TypeError:
            raise ValueError(
                "Could not serialize current LLM profile to JSON for update.")

        prompt_template = Prompts.get_intelligent_profile_update_prompt()
        formatted_prompt = prompt_template.format(
            original_profile_json=original_profile_json_str,
            new_information_text=new_information_text
        )

        try:
            llm_response = self.rag_system.llm.invoke(formatted_prompt)
            updated_profile_str = llm_response.content if hasattr(
                llm_response, 'content') else str(llm_response)

            # The prompt asks for JSON, so we parse it
            updated_llm_profile_dict = json.loads(updated_profile_str)

            # Basic validation: check if it's a dict
            if not isinstance(updated_llm_profile_dict, dict):
                raise ValueError(
                    "LLM did not return a valid dictionary structure for the updated profile.")

        except json.JSONDecodeError as e:
            # Log error e and the problematic string updated_profile_str
            raise ValueError(
                f"LLM did not return valid JSON for the profile update. Error: {e}")
        except Exception as e:
            # Log error e
            raise RuntimeError(f"Error invoking LLM for profile update: {e}")

        # Ensure all original keys are present if LLM omits some, or decide on merging strategy
        # For now, assume LLM returns the complete, updated profile structure.
        # A more robust merge:
        # merged_profile = current_llm_profile.copy()
        # merged_profile.update(updated_llm_profile_dict)
        # updated_llm_profile_dict = merged_profile

        return ProfileUpdateData(llm_profile=updated_llm_profile_dict)
