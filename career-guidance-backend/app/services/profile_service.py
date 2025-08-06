from pymongo.database import Database
from app.schemas.questionnaire import QuestionnaireSubmission, QuestionnaireInDB, ProfileUpdateData
from datetime import datetime
from bson import ObjectId
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProfileService:
    def __init__(self, mongo_db: Database):
        self.mongo_db = mongo_db

    def save_complete_questionnaire(self, data: QuestionnaireInDB) -> str:
        """
        Saves the complete questionnaire data (responses + LLM profile) to MongoDB.
        Updated method name to match the route usage.
        """
        print(data)
        collection = self.mongo_db["user_questionnaires"]

        # Convert to dict, MongoDB will generate _id
        data_to_insert = data.model_dump(exclude={"id"})

        result = collection.insert_one(data_to_insert)
        return str(result.inserted_id)

    def save_questionnaire_and_profile(self, data: QuestionnaireSubmission) -> str:
        """
        DEPRECATED: Use save_complete_questionnaire instead.
        """
        # This method is now deprecated since we're using QuestionnaireInDB directly
        raise NotImplementedError(
            "Use save_complete_questionnaire with QuestionnaireInDB instead")

    async def get_questionnaire_by_user_id(self, user_id: str) -> QuestionnaireInDB | None:
        """
        Retrieves the latest questionnaire data for a user from MongoDB.
        """
        collection = self.mongo_db["user_questionnaires"]
        document = collection.find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)])
        if document:
            document["id"] = str(document["_id"])  # Add string version of _id
            return QuestionnaireInDB(**document)
        return None

    async def update_user_profile(self, user_id: str, profile_update_data: ProfileUpdateData) -> bool:
        """
        Updates the llm_profile for a user's latest questionnaire entry in MongoDB.
        """
        collection = self.mongo_db["user_questionnaires"]

        # Find the latest document for the user
        latest_document = collection.find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)]
        )

        if not latest_document:
            # Or raise HTTPException(status_code=404, detail="Profile not found")
            return False

        # Update the llm_profile field
        result = collection.update_one(
            {"_id": latest_document["_id"]},
            {"$set": {"llm_profile": profile_update_data.llm_profile,
                      "timestamp": datetime.utcnow()}}  # Also update timestamp
        )
        # Return True if at least one document was modified.
        return result.modified_count > 0

    def get_user_questionnaire(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user questionnaire data with enhanced logging."""
        logger.info(
            f"🔍 ProfileService: Getting questionnaire for user_id: {user_id}")

        try:
            collection = self.mongo_db["user_questionnaires"]
            result = collection.find_one({"user_id": user_id})

            if result:
                logger.info(
                    f"✅ ProfileService: Found questionnaire data for user {user_id}")
                logger.info(f"📊 Data contains: {list(result.keys())}")

                # Log specific important fields
                if 'education_level' in result:
                    logger.info(
                        f"📚 Education level: {result['education_level']}")
                if 'llm_profile' in result:
                    logger.info(
                        f"🤖 LLM profile keys: {list(result['llm_profile'].keys())}")
                else:
                    logger.warning(
                        "⚠️ No LLM profile found in questionnaire data!")

                # Remove MongoDB's _id field for cleaner handling
                if "_id" in result:
                    del result["_id"]

                return result
            else:
                logger.warning(
                    f"❌ ProfileService: No questionnaire data found for user {user_id}")
                return None

        except Exception as e:
            logger.error(
                f"❌ ProfileService: Error retrieving questionnaire for user {user_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
