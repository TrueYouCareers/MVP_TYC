from fastapi import APIRouter, Depends, HTTPException ,  Path, Body
from app.db.connections import get_db_connections
from app.schemas.question import QuestionSchema, QuestionCreate
from app.services.question_service import QuestionService
from typing import List
from typing import Optional, Dict, Any


router = APIRouter()


@router.get("/{education_level}", response_model=List[QuestionSchema])
async def get_questions_by_education_level(education_level: str, db_clients=Depends(get_db_connections)):
    _, mongo = db_clients
    service = QuestionService(mongo)

    try:
        return await service.get_questions_by_level(education_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=QuestionSchema)
async def create_question(question_data: QuestionCreate, db_clients=Depends(get_db_connections)):
    _, mongo = db_clients
    service = QuestionService(mongo_db=mongo)

    try:
        return await service.add_question(question_data)
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{question_id}", response_model=QuestionSchema)
async def update_question_by_id(
    question_id: str = Path(..., description="MongoDB ObjectId of the question"),
    full_question: Dict = Body(..., description="Full updated question object"),
    db_clients=Depends(get_db_connections)
):
    _, mongo = db_clients
    service = QuestionService(mongo)

    try:
        updated_question = await service.update_question_by_id(question_id, full_question)
        return updated_question
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/{question_id}", status_code=200)
async def delete_question(
    question_id: str = Path(..., description="MongoDB ObjectId of the question"),
    db_clients=Depends(get_db_connections)
):
    _, mongo = db_clients
    service = QuestionService(mongo)

    try:
        deleted = await service.delete_question_by_id(question_id)

        if deleted:
            return {
                "status": "success",
                "message": "Question deleted successfully"
            }
        else:
            return {
                "status": "failure",
                "message": "Question not found"
            }

    except Exception as e:
        return {
            "status": "failure",
            "message": f"Deletion failed: {str(e)}"
        }