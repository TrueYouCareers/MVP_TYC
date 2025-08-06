# +++++++++++++++ THIS FILE IS REDUNDANT ++++++++++++++++

# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from typing import Dict, Any


# class GroqAgent:
#     def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
#         self.llm = ChatGroq(model_name=model_name, groq_api_key=api_key)

#     def generate_response(self, prompt: str) -> str:
#         response = self.llm.invoke(prompt)
#         return response.content if hasattr(response, 'content') else str(response)

#     def create_prompt(self, context: str, question: str, user_profile: Dict[str, Any]) -> PromptTemplate:
#         template = f"""
#         You are an expert career counselor with deep knowledge of the Indian education system, colleges, and job market.

#         Your task is to provide a helpful and informative answer based on the context and the user's profile.

#         Context: {context}

#         User Profile:
#         - Education Level: {user_profile.get('education_level', 'Unknown')}
#         - Interest Profile: {user_profile.get('profile', 'General')}
#         - Dominant Subjects: {user_profile.get('dominant_subjects', 'Various subjects')}

#         Question: {question}

#         Helpful Answer:
#         """
#         return PromptTemplate(template=template, input_variables=["context", "question"])
