from pymongo.database import Database
from pymongo.errors import PyMongoError
from app.schemas.question import QuestionCreate, QuestionSchema
from typing import List, Optional, Dict, Any
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)


class QuestionService:
    def __init__(self, mongo_db: Database):
        self.mongo_db = mongo_db
        self.collection = self.mongo_db["questions"]

    async def get_questions_by_level(self, education_level: str) -> List[QuestionSchema]:
        logger.info(f"📘 Fetching questions for level: {education_level}")
        try:
            cursor = self.collection.find({"education_level": education_level}).sort("order", 1)
            docs = cursor.to_list(length=None)

            questions = [QuestionSchema(**doc) for doc in docs]
            logger.info(f"✅ Found {len(questions)} questions for {education_level}")
            return questions
        except Exception as e:
            logger.error(f"❌ Error fetching questions: {str(e)}")
            raise



    async def add_question(self, question_data: QuestionCreate) -> QuestionSchema:
        logger.info(f"➕ Adding question: {question_data.name}")
        try:
            # Check for duplicate name
            existing = self.collection.find_one({"name": question_data.name})
            if existing:
                logger.warning(f"⚠️ Question with name '{question_data.name}' already exists.")
                raise ValueError(f"Question with name '{question_data.name}' already exists.")

            insert_result = self.collection.insert_one(question_data.dict())
            inserted = self.collection.find_one({"_id": insert_result.inserted_id})
            logger.info(f"✅ Successfully inserted question ID: {insert_result.inserted_id}")
            return QuestionSchema(**inserted)
        except Exception as e:
            logger.error(f"❌ Failed to insert question: {str(e)}")
            raise



    async def update_question_by_id(self, question_id: str, updated_data: dict) -> QuestionSchema:
            logger.info(f"✏️ Replacing entire question with ID: {question_id}")
            try:
                obj_id = ObjectId(question_id)

                # Ensure _id is not part of updated data (MongoDB restriction)
                updated_data.pop("id", None)
                updated_data.pop("_id", None)

                result = self.collection.replace_one({"_id": obj_id}, updated_data)

                if result.matched_count == 0:
                    raise ValueError("Question not found.")

                updated_doc = self.collection.find_one({"_id": obj_id})
                return QuestionSchema(**updated_doc)

            except Exception as e:
                logger.error(f"❌ Failed to update question ID {question_id}: {e}")
                raise

    async def delete_question_by_id(self, question_id: str) -> bool:
        try:
            obj_id = ObjectId(question_id)
            result = self.collection.delete_one({"_id": obj_id})

            if result.deleted_count == 1:
                logger.info(f"🗑️ Successfully deleted question ID: {question_id}")
                return True
            else:
                logger.warning(f"⚠️ No question found with ID: {question_id}")
                return False
        except PyMongoError as e:
            logger.error(f"❌ MongoDB deletion error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected deletion error: {e}")
            raise
