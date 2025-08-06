from typing import Dict, Any
from app.llm.rag_system import RAGSystem


class GuidanceService:
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self.rag_system = RAGSystem(knowledge_dir)

    def query_guidance(self, question: str, user_profile: Dict[str, Any], category: str = None) -> Dict[str, Any]:
        return self.rag_system.query(question, user_profile, category)
