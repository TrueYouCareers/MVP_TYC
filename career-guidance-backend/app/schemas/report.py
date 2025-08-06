from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportSection(BaseModel):
    title: str
    content: str  # Could be markdown or structured text
    visualizations: Optional[List[Dict[str, Any]]
                             ] = None  # For charts/graphs data


class ReportData(BaseModel):
    user_id: str
    report_title: str = "Personalized Career Guidance Report"
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Data derived from llm_profile
    profile_snapshot: Dict[str, Any]  # The LLM profile used for this report

    # Synthesized content for the report
    introduction: Optional[str] = None
    profile_analysis_summary: Optional[str] = None
    strengths_and_interests_detail: Optional[str] = None
    personality_insights_detail: Optional[str] = None

    # e.g., [{"path_name": "X", "analysis": "...", "skills": []}]
    career_path_explorations: List[Dict[str, Any]] = []

    # Summary of relevant chat interactions
    chat_session_recap: Optional[str] = None

    actionable_recommendations: List[str] = []
    concluding_remarks: Optional[str] = None

    # Raw sections if preferred by LLM output
    detailed_sections: Optional[List[ReportSection]] = None
