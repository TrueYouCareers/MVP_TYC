from bson import ObjectId
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler, BaseModel, Field
from typing import Optional, List, Literal, Dict

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This tells FastAPI how to represent it in OpenAPI docs
        return {'type': 'string'}



class QuestionSchema(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    education_level: Literal["10th", "12th", "graduate"]
    name: str
    title: str
    prompt: str
    type: Literal["single", "multi", "grouped", "text"]
    options: Optional[List[str]] = None
    grouped_options: Optional[Dict[str, List[str]]] = None
    depends_on: Optional[Dict[str, str]] = None
    is_required: Optional[bool] = True
    order: Optional[int] = 0

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}


class QuestionCreate(BaseModel):
    education_level: Literal["10th", "12th", "graduate"]
    name: str
    title: str
    prompt: str
    type: Literal["single", "multi", "grouped", "text"]
    options: Optional[List[str]] = None
    grouped_options: Optional[Dict[str, List[str]]] = None
    depends_on: Optional[Dict[str, str]] = None
    is_required: Optional[bool] = True
    order: Optional[int] = 0
