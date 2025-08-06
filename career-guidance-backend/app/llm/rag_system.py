from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # Import OpenAI for LLM fallback
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, List, Optional  # Added Optional
# Added for distance strategy
from langchain_community.vectorstores.utils import DistanceStrategy
import os
import json
import tiktoken
import logging  # Added logging
import tempfile  # Added tempfile for embedding fallback
import shutil  # Added shutil for embedding fallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants (similar to groq_guidance.py)
MAX_CONTEXT_TOKENS = 248000  # Example value, adjust as needed
RESERVED_TOKENS = 3000*2
MAX_CONTEXT_ALLOWED = MAX_CONTEXT_TOKENS - RESERVED_TOKENS
OPTIMAL_CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Ensure OPEN_AI_API_KEY is loaded
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

if not OPEN_AI_API_KEY:
    logger.error("OPEN_AI_API_KEY environment variable is not set.")
    # Depending on application design, you might raise an error or use a default key
    # raise ValueError("OPEN_AI_API_KEY environment variable is required")


class RAGSystem:
    def __init__(self, knowledge_dir="knowledge_base"):
        self.embeddings = self._initialize_embeddings_with_fallback()  # Use fallback mechanism
        self.llm = ChatOpenAI(
            model_name="gpt-4.1-mini",
            temperature=0.5,
            max_tokens=2048,
            max_retries=3,
            model_kwargs={"openai_api_key": OPEN_AI_API_KEY}
            )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.knowledge_dir = knowledge_dir
        self.vector_stores: Dict[str, FAISS] = {}
        self.merged_retriever: Optional[BaseRetriever] = None
        self.max_context_tokens = MAX_CONTEXT_ALLOWED

        if os.path.exists(knowledge_dir):
            self._initialize_vector_stores()
        else:
            logger.warning(
                f"Knowledge directory '{knowledge_dir}' not found. System will operate without a knowledge base.")
            # Not raising FileNotFoundError to allow operation without knowledge base initially

    def _initialize_vector_stores(self):
        logger.info("Initializing vector stores...")
        if not os.path.exists(self.knowledge_dir):
            logger.warning(
                f"Knowledge directory '{self.knowledge_dir}' does not exist. Cannot initialize vector stores.")
            return

        categories = [d for d in os.listdir(self.knowledge_dir) if os.path.isdir(
            os.path.join(self.knowledge_dir, d))]

        if not categories:
            logger.warning(
                f"No categories found in knowledge directory '{self.knowledge_dir}'.")
            return

        for category in categories:
            category_path = os.path.join(self.knowledge_dir, category)
            loader = DirectoryLoader(
                category_path, glob="**/*.txt",  # Recursive glob
                loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
                recursive=True,
                show_progress=True,
                use_multithreading=True  # Added for potentially faster loading
            )
            try:
                documents = loader.load()
                if not documents:
                    logger.info(
                        f"No documents found in category '{category}'.")
                    continue
                logger.info(
                    f"Loaded {len(documents)} documents for category '{category}'")
            except Exception as e:
                logger.error(
                    f"Error loading documents for category '{category}': {e}")
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=OPTIMAL_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Added separators
            )
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                logger.warning(
                    f"No document chunks created for category '{category}' after splitting.")
                continue

            try:
                vector_store = FAISS.from_documents(
                    chunks, self.embeddings, distance_strategy=DistanceStrategy.COSINE)
                self.vector_stores[category] = vector_store
                logger.info(
                    f"Created vector store for '{category}' with {len(chunks)} chunks")
            except Exception as e:
                logger.error(
                    f"Error creating vector store for category '{category}': {e}")

        self._create_merged_retriever()

    def relevance_score_fn(self, distance):
        # For cosine distance in [0, 2], similarity = 1 - (distance / 2)
        return 1 - (distance / 2)

    def _create_merged_retriever(self):
        """Create a merged retriever from all vector stores."""
        if not self.vector_stores:
            logger.warning(
                "No vector stores available to create a merged retriever.")
            self.merged_retriever = None  # Ensure it's None
            return

        try:
            from langchain.retrievers import EnsembleRetriever  # Ensure import

            retrievers = []
            weights = []  # Initialize weights

            for vs in self.vector_stores.values():

                retrievers.append(vs.as_retriever(
                    search_type="similarity_score_threshold",
                    # Optimized params
                    # Added relevance score function
                    search_kwargs={"k": 4, "score_threshold": 0.4,
                                   "relevance_score_fn": self.relevance_score_fn}
                ))
                # Assign equal weights, can be adjusted based on category importance
                weights.append(1.0 / len(self.vector_stores))

            if not retrievers:
                logger.warning(
                    "No individual retrievers created. Cannot form ensemble.")
                self.merged_retriever = None
                return

            if len(retrievers) == 1:
                self.merged_retriever = retrievers[0]
                logger.info(
                    "Single vector store found. Using its retriever as merged retriever.")
            else:
                self.merged_retriever = EnsembleRetriever(
                    retrievers=retrievers,
                    weights=weights
                )
                logger.info(
                    f"Ensemble retriever created with {len(retrievers)} sub-retrievers.")
        except ImportError:
            logger.warning(
                "langchain.retrievers.EnsembleRetriever not found. Using first available retriever as fallback.")
            if self.vector_stores:
                first_category = list(self.vector_stores.keys())[0]
                self.merged_retriever = self.vector_stores[first_category].as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 4, "score_threshold": 0.4}
                )
            else:
                self.merged_retriever = None
        except Exception as e:
            logger.error(f"Error creating merged retriever: {e}")
            # Fallback to a simple retriever if ensemble creation fails
            if self.vector_stores:
                first_category = list(self.vector_stores.keys())[0]
                self.merged_retriever = self.vector_stores[first_category].as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 4, "score_threshold": 0.4}
                )
                logger.info(
                    "Fell back to using the first available category retriever.")
            else:
                self.merged_retriever = None

    # Added Optional type hints
    def get_retriever(self, category: Optional[str] = None) -> Optional[BaseRetriever]:
        if category and category in self.vector_stores:
            logger.info(f"Using retriever for specific category: {category}")
            return self.vector_stores[category].as_retriever(
                search_type="similarity_score_threshold",  # More advanced search
                # Adjusted k and threshold
                search_kwargs={"k": 6, "score_threshold": 0.3}
            )
        elif not category:
            if self.merged_retriever is None and self.vector_stores:  # Check if vector_stores exist
                logger.info(
                    "Merged retriever not initialized. Attempting to create now.")
                self._create_merged_retriever()  # Attempt to create if not already

            if self.merged_retriever:
                logger.info("Using merged retriever for all categories.")
                return self.merged_retriever
            else:
                logger.warning(
                    "No merged retriever available and no specific category requested.")
                return None  # Return None if no retriever can be provided
        else:
            logger.error(f"Category '{category}' not found in knowledge base.")
            # raise ValueError(f"Category '{category}' not found in knowledge base") # Or return None
            return None

    # Added stream
    def query(self, question: str, user_profile: Dict[str, Any], chat_history: Optional[List[Dict[str, Any]]] = None, category: Optional[str] = None, stream: bool = False):
        # Enhanced logging for debugging profile context issues
        logger.info(f"=== RAG QUERY DEBUG INFO ===")
        logger.info(f"Question: {question[:100]}...")
        logger.info(f"Category: {category if category else 'all'}")
        logger.info(
            f"User Profile Keys: {list(user_profile.keys()) if user_profile else 'EMPTY PROFILE'}")

        # Detailed profile logging
        if user_profile:
            logger.info(
                f"Education Level: {user_profile.get('education_level', 'NOT SET')}")
            logger.info(
                f"Profile Summary: {user_profile.get('profile_summary', 'NOT SET')[:100]}...")
            logger.info(
                f"Keywords: {user_profile.get('identified_keywords', 'NOT SET')}")
            logger.info(
                f"Primary Orientation: {user_profile.get('primary_orientation', 'NOT SET')}")
        else:
            logger.warning(
                "❌ USER PROFILE IS EMPTY OR NONE - This will cause context issues!")

        logger.info(f"=== END DEBUG INFO ===")

        retriever = self.get_retriever(category)
        if not retriever:
            logger.warning("No retriever available for the query.")
            return {
                "answer": "Hello! I'm here to help with your career guidance. I notice I don't have your profile information yet. Could you tell me about your education level and interests so I can provide better personalized advice?",
                "source_documents": [],
                "token_usage": {"template": 0, "context": 0, "total_input": 0}
            }

        try:
            retrieved_docs = retriever.invoke(question)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            retrieved_docs = []

        prompt_template_obj = self._prepare_context_and_prompt(
            user_profile, chat_history)  # Changed to _prepare_context_and_prompt

        # Token counting and context optimization
        template_tokens = self._count_tokens(
            prompt_template_obj.template.replace(
                "{context}", "").replace("{question}", question)
        )
        context_text = self._optimize_context(
            retrieved_docs, question, user_profile)
        context_tokens = self._count_tokens(context_text)

        formatted_prompt = prompt_template_obj.format(
            context=context_text, question=question)
        total_tokens = template_tokens + context_tokens

        logger.info(
            f"Token usage - Template: {template_tokens}, Context: {context_tokens}, Total: {total_tokens}")

        try:
            if stream:
                logger.info("Streaming response.")
                return {
                    "answer_stream": self.llm.stream(formatted_prompt),
                    "source_documents": retrieved_docs,
                    "token_usage": {
                        "template": template_tokens,
                        "context": context_tokens,
                        "total_input": total_tokens
                    }
                }
            else:
                logger.info("Generating non-streamed response.")
                llm_response = self.llm.invoke(formatted_prompt)
                answer = llm_response.content if hasattr(
                    llm_response, 'content') else str(llm_response)
                return {
                    "answer": answer,
                    "source_documents": retrieved_docs,
                    "token_usage": {
                        "template": template_tokens,
                        "context": context_tokens,
                        "total_input": total_tokens
                    }
                }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "answer": "I encountered an issue while trying to generate a response. Please try again.",
                "source_documents": retrieved_docs,  # Still return retrieved docs if any
                "token_usage": {"template": template_tokens, "context": context_tokens, "total_input": total_tokens}
            }

    # Renamed and returns PromptTemplate
    # Renamed and updated
    def _prepare_context_and_prompt(self, user_profile: Dict[str, Any], chat_history: Optional[List[Dict[str, Any]]] = None) -> PromptTemplate:
        """Enhanced prompt preparation with better profile handling and logging."""

        # Log what we're working with
        logger.info(f"🔍 Preparing prompt with profile: {user_profile}")

        # Enhanced profile extraction with fallbacks and logging
        profile_summary = user_profile.get("profile_summary", "")
        identified_keywords_list = user_profile.get("identified_keywords", [])
        identified_keywords = ", ".join(
            identified_keywords_list) if identified_keywords_list else ""
        primary_orientation = user_profile.get("primary_orientation", "")
        education_level = user_profile.get("education_level", "").lower()

        # Log extracted values for debugging
        logger.info(f"📊 Extracted Profile Data:")
        logger.info(f"  - Education Level: '{education_level}'")
        logger.info(f"  - Profile Summary: '{profile_summary[:50]}...'")
        logger.info(f"  - Keywords: '{identified_keywords}'")
        logger.info(f"  - Primary Orientation: '{primary_orientation}'")

        # Handle missing profile data gracefully
        profile_available = bool(
            profile_summary or identified_keywords or primary_orientation or education_level)

        if not profile_available:
            logger.warning(
                "⚠️ NO PROFILE DATA AVAILABLE - Using generic context")
            user_profile_context = """
STUDENT PROFILE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❗ PROFILE NOT AVAILABLE: Please ask the user about their education level, interests, and background to provide personalized guidance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            # Enhanced user profile context with structured information
            user_profile_context = f"""
STUDENT PROFILE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 Education Level: {user_profile.get("education_level", "Not specified")}
🎯 Profile Summary: {profile_summary if profile_summary else "Not available"}
💡 Key Interests: {identified_keywords if identified_keywords else "Not specified"}
🧭 Primary Orientation: {primary_orientation if primary_orientation else "Not determined"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Enhanced base instructions with profile awareness
        base_instructions = """
🎓 ROLE: You are an expert AI career counselor with extensive experience guiding Indian students. You combine deep empathy with practical expertise.

🎯 MISSION: Help students discover their authentic career paths through personalized, actionable guidance rooted in Indian education systems and job markets.

⚠️ PROFILE AWARENESS: 
- If student profile is available, use it to provide highly personalized advice
- If profile is missing or limited, acknowledge this and ask clarifying questions
- ALWAYS be transparent about what information you have about the student

💬 COMMUNICATION STYLE:
- Warm, encouraging, and genuinely interested in the student's success
- Use a conversational tone that feels like talking to a wise mentor
- Keep responses concise and focused (2-4 short paragraphs maximum)
- Ask thoughtful follow-up questions to deepen understanding
- Acknowledge the student's emotions and concerns
- Use relevant examples when helpful

🧠 APPROACH:
1. Always start by acknowledging the student's current situation
2. Connect your advice directly to their profile and interests (if available)
3. If profile data is missing, ask specific questions to gather it
4. Provide specific, actionable next steps
5. Balance optimism with realistic expectations
6. End with 1-2 thoughtful questions to continue the conversation

📊 CONTEXT USAGE:
- Prioritize retrieved context that aligns with the student's profile
- If context is limited, draw from general knowledge but stay relevant
- Always indicate when you're unsure and suggest additional resources

IMPORTANT: Keep your response under 200 words generally. If a deep question has been raised, it's okay to go over and explore it further. Be concise, actionable, and conversational.
"""

        # Education-level specific guidance with enhanced Indian context
        education_specific_guidance_map = {
            "10th": """
🎯 FOCUS FOR 10TH STUDENTS (Keep under 200 words):
- Stream selection (Science/Commerce/Arts) based on genuine interests, not just marks.
- Bust common myths: "Only Science leads to good careers."
- Explain how each stream connects to modern, high-paying careers.
- Address family pressure with practical solutions.
- Use age-appropriate language for 14-15 year olds.

KEY AREAS: Stream counseling, interest exploration, family communication strategies.
""",
            "12th": """
🎯 FOCUS FOR 12TH STUDENTS (Keep under 200 words):
- Specific degree choices and entrance exam strategies.
- Connect 11th/12th subjects to undergraduate courses and careers.
- Discuss emerging fields and traditional paths equally.
- Address entrance exam stress and backup plans.
- Mention specific Indian universities and admission processes.

KEY AREAS: Course selection, entrance exams (JEE/NEET/CLAT/CUET), university guidance.
""",
            "graduate": """
🎯 FOCUS FOR GRADUATES (Keep under 200 words):
- Bridge the gap between degree and industry requirements.
- Emphasize skill-based career transitions.
- Address "degree vs. skills" concerns.
- Provide concrete upskilling roadmaps.
- Discuss corporate readiness and interview preparation.

KEY AREAS: Skill development, career transitions, industry readiness, postgrad options.
"""
        }

        specific_guidance = education_specific_guidance_map.get(
            # Default to graduate
            education_level, education_specific_guidance_map["graduate"])

        chat_history_context = ""
        if chat_history:
            # Take last few messages to avoid large context
            # Last 3 pairs of user/ai messages
            history_to_include = chat_history[-6:]
            formatted_history = []
            for message in history_to_include:
                role = str(message.get("role", "unknown")).capitalize()
                content = str(message.get("content", "")).strip()
                if content:  # Avoid adding empty messages
                    formatted_history.append(f"{role}: {content}")

            if formatted_history:
                chat_history_str = "\n".join(formatted_history)
                chat_history_context = f"""
📜 RECENT CONVERSATION HISTORY (for context):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chat_history_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Enhanced template with better structure and flow
        template_string = f"""{base_instructions}

{user_profile_context}

{specific_guidance}

{chat_history_context}
📖 RETRIEVED KNOWLEDGE:
{{context}}

💭 STUDENT'S CURRENT QUESTION: {{question}}

🗣️ YOUR RESPONSE:
Keep your response under 200 words. Structure it as:
[Brief acknowledgment] → [2-3 key points with specific advice] → [1-2 follow-up questions]

Response:
"""
        return PromptTemplate.from_template(template_string)

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(
                f"Token counting error: {e}. Falling back to char/4 estimation.")
            return len(text) // 4  # Fallback

    def _optimize_context(self, retrieved_docs: List[Document], question: str, user_profile: Dict[str, Any]) -> str:
        """Optimizes context to fit within token limits, prioritizing relevance."""
        if not retrieved_docs:
            return "No specific context retrieved. Please rely on general knowledge."

        # Simple relevance: use the first few documents, assuming they are most relevant.
        # A more advanced implementation could re-rank documents based on question and user_profile.

        context_parts = []
        current_tokens = 0
        # Max tokens for context, reserving space for prompt template, question, and response generation
        # This is a rough heuristic; adjust as needed.
        max_context_tokens_for_prompt = MAX_CONTEXT_ALLOWED - \
            self._count_tokens(question) - \
            500  # Reserve 500 for prompt and response buffer

        for doc in retrieved_docs[:5]:  # Consider top 5 documents for context
            doc_text = doc.page_content
            doc_tokens = self._count_tokens(doc_text)

            if current_tokens + doc_tokens <= max_context_tokens_for_prompt:
                context_parts.append(doc_text)
                current_tokens += doc_tokens
            else:
                # Try to add a truncated version of the document if it's too long
                remaining_tokens = max_context_tokens_for_prompt - current_tokens
                if remaining_tokens > 50:  # Only add if a meaningful part can be included
                    truncated_text = self._truncate_to_tokens(
                        doc_text, remaining_tokens)
                    context_parts.append(truncated_text)
                    current_tokens += self._count_tokens(truncated_text)
                break  # Stop adding documents once limit is reached or exceeded

        if not context_parts:
            return "Retrieved context was too long to fit. Please rely on general knowledge or rephrase your query."

        return "\n\n---\n\n".join(context_parts)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to approximately fit within a token limit."""
        # This is a rough approximation. For precise truncation, one would tokenize and then join.
        # Average token length can vary, using 4 chars/token as a heuristic.
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated_text = text[:max_chars]
        # Try to end on a sentence boundary for readability
        last_period = truncated_text.rfind('.')
        if last_period > max_chars * 0.75:  # If period is in the last 25%
            return truncated_text[:last_period + 1]

        # If no good sentence boundary, just truncate and add ellipsis
        # Remove last partial word
        return truncated_text.rsplit(' ', 1)[0] + "..."

    def _initialize_embeddings_with_fallback(self):
        """Initializes HuggingFace embeddings with fallback mechanisms."""
        # Attempt 1: Standard model
        primary_model = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            logger.info(
                f"Attempting to initialize embeddings with model: {primary_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=primary_model,
                encode_kwargs={'normalize_embeddings': True}
            )
            embeddings.embed_query("Test query")  # Test the embedding
            logger.info(
                f"Successfully initialized embeddings with {primary_model}")
            return embeddings
        except Exception as e:
            logger.warning(
                f"Failed to initialize {primary_model}: {e}. Trying temporary cache.")

        # Attempt 2: Use temporary cache directory
        try:
            logger.info(
                f"Attempting to initialize {primary_model} with temporary cache...")
            with tempfile.TemporaryDirectory(prefix="hf_cache_") as temp_cache_dir:
                os.environ['HF_HOME'] = temp_cache_dir
                os.environ['TRANSFORMERS_CACHE'] = os.path.join(
                    temp_cache_dir, 'transformers')
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(
                    temp_cache_dir, 'sentence_transformers')

                embeddings = HuggingFaceEmbeddings(
                    model_name=primary_model,
                    cache_folder=temp_cache_dir,  # Explicitly set cache_folder
                    encode_kwargs={'normalize_embeddings': True}
                )
                embeddings.embed_query("Test query")
                logger.info(
                    f"Successfully initialized {primary_model} with temporary cache: {temp_cache_dir}")
                # Note: temp_cache_dir will be cleaned up after this block,
                # so models might be re-downloaded on next run unless persisted.
                # For persistent temp cache, manage directory lifecycle outside 'with'.
                return embeddings
        except Exception as e:
            logger.warning(
                f"Failed to initialize {primary_model} with temporary cache: {e}. Trying alternative model.")

        # Attempt 3: Alternative smaller model
        alternative_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        try:
            logger.info(
                f"Attempting to initialize embeddings with alternative model: {alternative_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=alternative_model,
                encode_kwargs={'normalize_embeddings': True}
            )
            embeddings.embed_query("Test query")
            logger.info(
                f"Successfully initialized embeddings with {alternative_model}")
            return embeddings
        except Exception as e:
            logger.error(
                f"Failed to initialize {alternative_model}: {e}. Critical embedding failure.")
            logger.warning(
                "Using a very basic fallback embedding. Search quality will be severely impacted.")
            return self._create_very_basic_fallback_embeddings()

    def _create_very_basic_fallback_embeddings(self):
        """Creates a very simple fallback embedding class if all else fails."""
        class BasicFallbackEmbeddings:
            def embed_query(self, text: str) -> List[float]:
                import hashlib
                # Simple hash-based embedding, very rudimentary
                val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                # Create a small, fixed-size vector
                return [float((val >> (i*8)) & 0xFF) / 255.0 for i in range(16)]

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self.embed_query(text) for text in texts]

        logger.critical(
            "Initialized with basic fallback embeddings. Functionality will be limited.")
        return BasicFallbackEmbeddings()

    def _analyze_response_patterns(self, formatted_responses: str, education_level: str) -> str:
        """
        Performs a basic analysis of user responses to provide guiding context to the LLM.
        """
        num_questions_answered = formatted_responses.count("Q:")

        analysis_summary = (
            f"Preliminary analysis of {num_questions_answered} responses for a {education_level} student:\n"
            f"The student has provided responses to {num_questions_answered} questions. "
            f"These responses cover various aspects of their interests, preferences, and academic background. "
            f"When generating the profile, focus on identifying dominant themes, inherent strengths, and potential areas for exploration based on these answers. "
            f"Consider patterns in their choices and any explicitly stated preferences or concerns."
        )
        if num_questions_answered == 0:
            analysis_summary = "No specific responses provided by the student. Generate a general profile based on education level, focusing on common patterns and exploratory advice."

        return analysis_summary

    def generate_llm_student_profile(self, education_level: str, responses: Dict[str, str], questions_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced LLM profile generation with concise, keyword-focused output"""
        # Handle case where questions_data is not provided
        if questions_data is None:
            questions_data = {}
            logger.warning(
                "No questions_data provided, using response keys only")

        formatted_responses_lines = []
        for q_id, answer_key in responses.items():
            if q_id in questions_data:
                question_info = questions_data[q_id]
                question_text = question_info["text"]
                answer_text = question_info["options"].get(answer_key, "N/A")
                formatted_responses_lines.append(
                    f"Q: {question_text}\nA: ({answer_key}) {answer_text}")
            else:
                # Fallback when questions_data is not available
                formatted_responses_lines.append(
                    f"Q: Question {q_id}\nA: {answer_key}")

        formatted_responses_str = "\n\n".join(formatted_responses_lines)

        # Perform basic analysis of responses
        analyzed_patterns = self._analyze_response_patterns(
            formatted_responses_str, education_level)

        # Updated prompt for concise, visual-friendly output
        prompt_template = """
You are Dr. Priya Sharma, an expert career psychologist specializing in concise student assessment.

CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no markdown, no code blocks - just pure JSON.

TASK: Create a CONCISE personality profile for a {education_level} student. Keep text brief and focus on keywords.

STUDENT RESPONSES:
{formatted_responses}

PRELIMINARY ANALYSIS OF RESPONSES:
{analyzed_patterns}

INSTRUCTIONS:
1. Analyze response patterns quickly - focus on dominant themes based on the STUDENT RESPONSES and PRELIMINARY ANALYSIS.
2. Keep all text SHORT (1-2 sentences max)
3. Emphasize keywords and natural inclinations
4. Provide specific, actionable recommendations
5. Include data for visual representation

REQUIRED OUTPUT FORMAT - EXACTLY THIS STRUCTURE:
{{
  "profile_summary": "1-2 sentences about natural inclination and current position. Be specific but brief.",
  "identified_keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
  "primary_orientation": "One word: Analytical/Creative/Social/Practical/Investigative/Enterprising",
  "orientation_confidence": 85,
  "personality_scores": {{
    "analytical": 75,
    "creative": 60,
    "social": 45,
    "practical": 80,
    "investigative": 70
  }},
  "top_strength": "Most dominant trait in 1-2 words",
  "learning_style": "Visual/Auditory/Kinesthetic/Reading",
  "potential_career_paths": [
    {{"path": "Career Title 1", "match_percentage": 90}},
    {{"path": "Career Title 2", "match_percentage": 75}},
    {{"path": "Career Title 3", "match_percentage": 65}}
  ],
  "recommended_next_steps": [
    "Specific step 1",
    "Specific step 2"
  ],
  "confidence_indicators": {{
    "decision_making": "High/Medium/Low",
    "self_awareness": "High/Medium/Low",
    "exploration_readiness": "High/Medium/Low"
  }},
  "interest_distribution": {{
    "STEM": 40,
    "Arts_Humanities": 20,
    "Business_Commerce": 25,
    "Social_Services": 15
  }}
}}

CRITICAL REQUIREMENTS:
- Profile summary: Maximum 2 sentences
- Use exact percentage numbers for scores (0-100)
- All text fields should be concise
- Focus on natural inclinations and current readiness
- Respond with ONLY the JSON object
"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Generate the profile using the LLM with better error handling and debugging
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating LLM profile (attempt {attempt + 1})")

                # Use a more constrained LLM call
                llm_response = self.llm.invoke(
                    prompt.format(
                        education_level=education_level,
                        formatted_responses=formatted_responses_str,
                        analyzed_patterns=analyzed_patterns
                    )
                )

                # Extract content and clean it
                if hasattr(llm_response, 'content'):
                    json_str = llm_response.content.strip()
                else:
                    json_str = str(llm_response).strip()

                # Log the raw response for debugging
                logger.info(f"Raw LLM response length: {len(json_str)}")
                logger.info(f"Raw LLM response preview: {json_str[:200]}...")

                # Check if response is empty
                if not json_str:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

                # Clean up potential formatting issues
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]

                # Remove any leading/trailing whitespace or newlines
                json_str = json_str.strip()

                # Ensure it starts and ends with braces
                if not json_str.startswith('{'):
                    # Try to find the first { and last }
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx+1]
                    else:
                        logger.warning(
                            f"No valid JSON structure found on attempt {attempt + 1}")
                        continue

                # Attempt to parse JSON
                profile_data = json.loads(json_str)

                # Validate required fields
                required_fields = ["profile_summary", "identified_keywords", "primary_orientation",
                                   "potential_career_paths", "recommended_next_steps", "personality_scores"]

                missing_fields = [
                    field for field in required_fields if field not in profile_data]
                if missing_fields:
                    logger.warning(
                        f"Missing required fields: {missing_fields} on attempt {attempt + 1}")
                    continue

                logger.info("LLM profile generated successfully")
                return profile_data

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parsing error on attempt {attempt + 1}: {e}")
                logger.warning(f"Problematic JSON string: {json_str[:500]}...")
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to generate valid JSON after all attempts")

            except Exception as e:
                logger.error(
                    f"Error generating LLM profile on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break

        # Enhanced fallback profile with visual data
        logger.info("Using fallback profile due to LLM generation failure")
        return {
            "profile_summary": f"Shows balanced interests across multiple areas. Ready for focused exploration to identify strongest career direction.",
            "identified_keywords": ["Balanced", "Exploring", "Adaptable", "Learning", "Growing"],
            "primary_orientation": "Exploratory",
            "orientation_confidence": 60,
            "personality_scores": {
                "analytical": 50,
                "creative": 50,
                "social": 50,
                "practical": 50,
                "investigative": 50
            },
            "top_strength": "Adaptability",
            "learning_style": "Mixed",
            "potential_career_paths": [
                {"path": "Multiple options available", "match_percentage": 70},
                {"path": "Skill-based development", "match_percentage": 65},
                {"path": "Consultation recommended", "match_percentage": 60}
            ],
            "recommended_next_steps": [
                "Take detailed aptitude assessment",
                "Explore internships in interested fields"
            ],
            "confidence_indicators": {
                "decision_making": "Medium",
                "self_awareness": "Medium",
                "exploration_readiness": "High"
            },
            "interest_distribution": {
                "STEM": 25,
                "Arts_Humanities": 25,
                "Business_Commerce": 25,
                "Social_Services": 25
            }
        }

    def generate_llm_student_profile_12(self, education_level: str, responses: Dict[str, str], questions_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a structured student personality profile including Class 12th style report fields."""

        if questions_data is None:
            questions_data = {}
            logger.warning("No questions_data provided, using response keys only")

        formatted_responses_lines = []
        for q_id, answer_key in responses.items():
            if q_id in questions_data:
                question_info = questions_data[q_id]
                question_text = question_info["text"]
                answer_text = question_info["options"].get(answer_key, "N/A")
                formatted_responses_lines.append(f"Q: {question_text}\nA: ({answer_key}) {answer_text}")
            else:
                formatted_responses_lines.append(f"Q: Question {q_id}\nA: {answer_key}")

        formatted_responses_str = "\n\n".join(formatted_responses_lines)
        analyzed_patterns = self._analyze_response_patterns(formatted_responses_str, education_level)

        # Strong and structured prompt
        prompt_template = """
    You are Dr. Priya Sharma, a top-tier career psychologist who crafts professional and concise student profiles.

    REQUIREMENT: Respond with ONLY raw JSON. No markdown, no explanations. The JSON should be fully parseable and match the structure exactly.

    OBJECTIVE: Generate a brief, keyword-rich profile for a {education_level} student. Use a friendly and motivational tone. Keep each field very short and high impact (1-2 lines).

    DATA:
    STUDENT RESPONSES:
    {formatted_responses}

    PRELIMINARY ANALYSIS:
    {analyzed_patterns}

    REQUIRED JSON STRUCTURE:
    {{
    "profile_summary": "Brief summary of student's overall traits and inclination.",
    "identified_keywords": ["Use 5 high-impact career-related keywords"],
    "primary_orientation": "One of: Analytical / Creative / Social / Practical / Investigative / Enterprising",
    "orientation_confidence": 85,
    "personality_scores": {{
        "analytical": 0–100,
        "creative": 0–100,
        "social": 0–100,
        "practical": 0–100,
        "investigative": 0–100
    }},
    "top_strength": "Most dominant quality (e.g. Problem-Solving)",
    "learning_style": "One of: Visual / Auditory / Kinesthetic / Reading",
    "potential_career_paths": [
        {{"path": "Career Title 1", "match_percentage": 90}},
        {{"path": "Career Title 2", "match_percentage": 75}},
        {{"path": "Career Title 3", "match_percentage": 65}}
    ],
    "recommended_next_steps": [
        "Action-oriented and practical step 1",
        "Action-oriented and practical step 2"
    ],
    "confidence_indicators": {{
        "decision_making": "High / Medium / Low",
        "self_awareness": "High / Medium / Low",
        "exploration_readiness": "High / Medium / Low"
    }},
    "interest_distribution": {{
        "STEM": 0–100,
        "Arts_Humanities": 0–100,
        "Business_Commerce": 0–100,
        "Social_Services": 0–100
    }},
    "welcome_statement": "A warm and motivational message that acknowledges the student's effort and journey.",
    "your_natural_inclination": "Summary of what kind of domains or tasks the student naturally enjoys.",
    "potential_career_options": [
        "Mention 1st career option that align with current interests and mindset, with quick reasoning.",
        "Mention 2nd career option that align with current interests and mindset, with quick reasoning.",
        "Or more"
    ],
    "your_strengths_and_qualities": [
        "Brief praise highlighting strong quality (e.g. creativity, focus, curiosity).",
        "Brief praise highlighting strong quality (e.g. creativity, focus, curiosity).",
        "Or more"
        ],
    "possible_roadblocks": [
        "Possible risks, habits, or mindsets that may slow progress, with soft advice.",
        "Possible risks, habits, or mindsets that may slow progress, with soft advice.",
        "Or more"
    ],
    "remarks": "Mentor-like advice with a positive tone about the next steps and general encouragement.",
    "profile_in_a_gist": {{
        "career_paths": ["List of 2 possible careers"],
        "strengths": ["List of 2 notable traits"],
        "exams_to_consider": ["Relevant entrance exams, if any"],
        "roadblock": "Name the biggest challenge right now",
        "suggestion": "Concrete suggestion – talk to mentor, take test, explore internship, etc.",
        "final_note": "A strong concluding sentence – inspiring and reassuring."
    }}
    }}
    """

        prompt = PromptTemplate.from_template(prompt_template)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating LLM profile (attempt {attempt + 1})")

                llm_response = self.llm.invoke(
                    prompt.format(
                        education_level=education_level,
                        formatted_responses=formatted_responses_str,
                        analyzed_patterns=analyzed_patterns
                    )
                )

                json_str = getattr(llm_response, 'content', str(llm_response)).strip()

                if not json_str:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                elif json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]

                json_str = json_str.strip()

                if not json_str.startswith('{'):
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx + 1]
                    else:
                        logger.warning(f"No valid JSON structure found on attempt {attempt + 1}")
                        continue

                profile_data = json.loads(json_str)

                required_fields = [
                    "profile_summary", "identified_keywords", "primary_orientation", "orientation_confidence",
                    "personality_scores", "top_strength", "learning_style", "potential_career_paths",
                    "recommended_next_steps", "confidence_indicators", "interest_distribution",
                    "welcome_statement", "your_natural_inclination", "potential_career_options",
                    "your_strengths_and_qualities", "possible_roadblocks", "remarks", "profile_in_a_gist"
                ]

                missing_fields = [field for field in required_fields if field not in profile_data]
                if missing_fields:
                    logger.warning(f"Missing fields: {missing_fields} on attempt {attempt + 1}")
                    continue

                logger.info("LLM profile generated successfully")
                return profile_data

            except json.JSONDecodeError as e:
                logger.warning(f"JSON error on attempt {attempt + 1}: {e}")
                logger.warning(f"Problematic JSON string: {json_str[:500]}...")
                if attempt == max_retries - 1:
                    logger.error("Failed to generate valid JSON after all attempts")

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break

        logger.info("Using fallback profile due to LLM failure")
        return {
            "profile_summary": "You have diverse interests and are open to discovering new paths. A great sign of a flexible learner.",
            "identified_keywords": ["Curious", "Adaptable", "Multi-domain", "Reflective", "Learning-oriented"],
            "primary_orientation": "Exploratory",
            "orientation_confidence": 60,
            "personality_scores": {
                "analytical": 50,
                "creative": 55,
                "social": 50,
                "practical": 60,
                "investigative": 50
            },
            "top_strength": "Adaptability",
            "learning_style": "Mixed",
            "potential_career_paths": [
                {"path": "Generalist Consultant", "match_percentage": 70},
                {"path": "Business Analyst", "match_percentage": 65},
                {"path": "Multidisciplinary Researcher", "match_percentage": 60}
            ],
            "recommended_next_steps": [
                "Take a deep-dive career interest test",
                "Shadow professionals in 2 areas you find exciting"
            ],
            "confidence_indicators": {
                "decision_making": "Medium",
                "self_awareness": "Medium",
                "exploration_readiness": "High"
            },
            "interest_distribution": {
                "STEM": 25,
                "Arts_Humanities": 25,
                "Business_Commerce": 25,
                "Social_Services": 25
            },
            "welcome_statement": "Welcome! This report is your first step to understanding your potential and direction.",
            "your_natural_inclination": "You are inclined towards exploring multiple disciplines with a mix of creativity and logic.",
            "potential_career_options": "Consulting, interdisciplinary research, or exploratory business roles may suit you well.",
            "your_strengths_and_qualities": "You’re versatile, open-minded, and quick to adapt to new information or environments.",
            "possible_roadblocks": "Might struggle with choosing one path—don’t let indecision slow you down. Seek clarity through action.",
            "remarks": "This is just a beginning. Keep taking initiative, ask questions, and stay curious about yourself and the world.",
            "profile_in_a_gist": {
                "career_paths": ["Business Consultant", "Product Researcher"],
                "strengths": ["Adaptability", "Curiosity"],
                "exams_to_consider": ["Aptitude Mapping Test", "Stream Selector"],
                "roadblock": "Lack of certainty about next steps",
                "suggestion": "Talk to a career mentor and explore short-term online courses.",
                "final_note": "Your curiosity is a strength — keep building, learning, and evolving with it."
            }
        }
    
    def generate_llm_student_profile_10(self, education_level: str, responses: Dict[str, str], questions_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a structured student personality profile including Class 10th style report fields."""

        if questions_data is None:
            questions_data = {}
            logger.warning("No questions_data provided, using response keys only")

        formatted_responses_lines = []
        for q_id, answer_key in responses.items():
            if q_id in questions_data:
                question_info = questions_data[q_id]
                question_text = question_info["text"]
                answer_text = question_info["options"].get(answer_key, "N/A")
                formatted_responses_lines.append(f"Q: {question_text}\nA: ({answer_key}) {answer_text}")
            else:
                formatted_responses_lines.append(f"Q: Question {q_id}\nA: {answer_key}")

        formatted_responses_str = "\n\n".join(formatted_responses_lines)
        analyzed_patterns = self._analyze_response_patterns(formatted_responses_str, education_level)

        prompt_template = """
    You are Dr. Priya Sharma, a top-tier career psychologist who crafts professional and concise student profiles.

    REQUIREMENT: Respond with ONLY raw JSON. No markdown, no explanations. The JSON should be fully parseable and match the structure exactly.

    OBJECTIVE: Generate a brief, keyword-rich profile for a {education_level} student. Use a friendly and motivational tone. Keep each field very short and high impact (1-2 lines).

    DATA:
    STUDENT RESPONSES:
    {formatted_responses}

    PRELIMINARY ANALYSIS:
    {analyzed_patterns}

    REQUIRED JSON STRUCTURE:
    {{
    "profile_summary": "Brief summary of student's overall traits and inclination.",
    "identified_keywords": ["Use 5 high-impact career-related keywords"],
    "primary_orientation": "One of: Analytical / Creative / Social / Practical / Investigative / Enterprising",
    "orientation_confidence": 85,
    "personality_scores": {{
        "analytical": 0–100,
        "creative": 0–100,
        "social": 0–100,
        "practical": 0–100,
        "investigative": 0–100
    }},
    "top_strength": "Most dominant quality (e.g. Problem-Solving)",
    "learning_style": "One of: Visual / Auditory / Kinesthetic / Reading",
    "potential_career_paths": [
        {{"path": "Career Title 1", "match_percentage": 90}},
        {{"path": "Career Title 2", "match_percentage": 75}},
        {{"path": "Career Title 3", "match_percentage": 65}}
    ],
    "recommended_next_steps": [
        "Action-oriented and practical step 1",
        "Action-oriented and practical step 2"
    ],
    "confidence_indicators": {{
        "decision_making": "High / Medium / Low",
        "self_awareness": "High / Medium / Low",
        "exploration_readiness": "High / Medium / Low"
    }},
    "interest_distribution": {{
        "STEM": 0–100,
        "Arts_Humanities": 0–100,
        "Business_Commerce": 0–100,
        "Social_Services": 0–100
    }},
    "welcome_statement": "A warm and encouraging note to acknowledge the student’s efforts and this milestone step.",
    "your_natural_inclination": "Subjects you enjoy and the natural pull you feel toward a specific domain or interest area.",
    "potential_career_options": [
        "1st possible career based on your early interests and strengths.",
        "2nd possible career based on your early interests and strengths.",
        "Or more"
    ],
    "your_strengths_and_qualities": [
        "Brief praise highlighting strong quality (e.g. creativity, focus, curiosity).",
        "Brief praise highlighting strong quality (e.g. creativity, focus, curiosity).",
        "Or more"
    ],
    "possible_roadblocks": [
        "Any confusion or hesitation you may be facing, with soft advice to navigate it.",
        "Any confusion or hesitation you may be facing, with soft advice to navigate it.",
        "Or more"
    ],
    "final_note": "You don’t need to have all the answers right now — stay curious, keep exploring, and trust the learning process.",
    "profile_in_a_gist": {{
        "subjects_good_at": ["Subjects where you perform well"],
        "natural_calling": "What you naturally gravitate towards doing or exploring",
        "inclined_to_pursue": ["Suggested career directions"],
        "roadblocks": ["Any roadblocks currently visible (like confusion, low exposure)"],
        "encouragement": "You're not expected to have all the answers today. Stay curious and keep learning!"
    }}
    }}
    """

        prompt = PromptTemplate.from_template(prompt_template)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating LLM profile (attempt {attempt + 1})")

                llm_response = self.llm.invoke(
                    prompt.format(
                        education_level=education_level,
                        formatted_responses=formatted_responses_str,
                        analyzed_patterns=analyzed_patterns
                    )
                )

                json_str = getattr(llm_response, 'content', str(llm_response)).strip()

                if not json_str:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                elif json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]

                json_str = json_str.strip()

                if not json_str.startswith('{'):
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx + 1]
                    else:
                        logger.warning(f"No valid JSON structure found on attempt {attempt + 1}")
                        continue

                profile_data = json.loads(json_str)

                required_fields = [
                    "profile_summary", "identified_keywords", "primary_orientation", "orientation_confidence",
                    "personality_scores", "top_strength", "learning_style", "potential_career_paths",
                    "recommended_next_steps", "confidence_indicators", "interest_distribution",
                    "welcome_statement", "your_natural_inclination", "potential_career_options",
                    "your_strengths_and_qualities", "possible_roadblocks", "final_note", "profile_in_a_gist"
                ]

                missing_fields = [field for field in required_fields if field not in profile_data]
                if missing_fields:
                    logger.warning(f"Missing fields: {missing_fields} on attempt {attempt + 1}")
                    continue

                logger.info("LLM profile generated successfully")
                return profile_data

            except json.JSONDecodeError as e:
                logger.warning(f"JSON error on attempt {attempt + 1}: {e}")
                logger.warning(f"Problematic JSON string: {json_str[:500]}...")
                if attempt == max_retries - 1:
                    logger.error("Failed to generate valid JSON after all attempts")

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break

        logger.info("Using fallback profile due to LLM failure")
        return {
            "profile_summary": "You show promise across different areas and are just beginning your journey of exploration.",
            "identified_keywords": ["Curious", "Open-minded", "Balanced", "Evolving", "Reflective"],
            "primary_orientation": "Exploratory",
            "orientation_confidence": 60,
            "personality_scores": {
                "analytical": 50,
                "creative": 50,
                "social": 50,
                "practical": 50,
                "investigative": 50
            },
            "top_strength": "Curiosity",
            "learning_style": "Mixed",
            "potential_career_paths": [
                {"path": "General Explorer", "match_percentage": 70},
                {"path": "Multi-domain Learner", "match_percentage": 65},
                {"path": "Interdisciplinary Thinker", "match_percentage": 60}
            ],
            "recommended_next_steps": [
                "Talk to a school counselor",
                "Explore 2 subjects through online projects or quizzes"
            ],
            "confidence_indicators": {
                "decision_making": "Medium",
                "self_awareness": "Medium",
                "exploration_readiness": "High"
            },
            "interest_distribution": {
                "STEM": 25,
                "Arts_Humanities": 25,
                "Business_Commerce": 25,
                "Social_Services": 25
            },
            "welcome_statement": "Welcome! This profile is the start of discovering your unique talents and interests.",
            "your_natural_inclination": "You are drawn to learning across subjects and show signs of both curiosity and creativity.",
            "potential_career_options": "Careers like design thinking, early entrepreneurship or creative writing may interest you.",
            "your_strengths_and_qualities": "You are curious, imaginative, and enjoy exploring topics in depth.",
            "possible_roadblocks": "Sometimes too many interests can confuse direction. Stay focused on what excites you most.",
            "remarks": "You’re doing well—explore slowly, reflect deeply, and ask questions. It’s your journey.",
            "profile_in_a_gist": {
                "subjects_good_at": ["Science", "English"],
                "natural_calling": "Exploring new ideas and expressing yourself creatively",
                "inclined_to_pursue": ["Creative Arts", "Research"],
                "roadblock": "Unsure which interest to focus on",
                "suggestion": "Speak to your teacher or counselor and pick a short course in your area of curiosity.",
                "final_note": "You’re not expected to have all the answers today — stay curious and keep learning!"
            }
        }
        
    def generate_llm_student_profile_graduate(self, education_level: str, responses: Dict[str, str], questions_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a structured graduate personality profile with extended graduation-specific fields."""

        if questions_data is None:
            questions_data = {}
            logger.warning("No questions_data provided, using response keys only")

        formatted_responses_lines = []
        for q_id, answer_key in responses.items():
            if q_id in questions_data:
                question_info = questions_data[q_id]
                question_text = question_info["text"]
                answer_text = question_info["options"].get(answer_key, "N/A")
                formatted_responses_lines.append(f"Q: {question_text}\nA: ({answer_key}) {answer_text}")
            else:
                formatted_responses_lines.append(f"Q: Question {q_id}\nA: {answer_key}")

        formatted_responses_str = "\n\n".join(formatted_responses_lines)
        analyzed_patterns = self._analyze_response_patterns(formatted_responses_str, education_level)

        prompt_template = """
    You are Dr. Priya Sharma, a top-tier career psychologist who crafts professional and concise student profiles.

    REQUIREMENT: Respond with ONLY raw JSON. No markdown, no explanations. The JSON should be fully parseable and match the structure exactly.

    OBJECTIVE: Generate a brief, keyword-rich profile for a {education_level} student. Use a friendly and motivational tone. Keep each field very short and high impact (1-2 lines).

    DATA:
    STUDENT RESPONSES:
    {formatted_responses}

    PRELIMINARY ANALYSIS:
    {analyzed_patterns}

    REQUIRED JSON STRUCTURE:
    {{
    "profile_summary": "Brief summary of student's overall traits and inclination.",
    "identified_keywords": ["Use 5 high-impact career-related keywords"],
    "primary_orientation": "One of: Analytical / Creative / Social / Practical / Investigative / Enterprising",
    "orientation_confidence": 85,
    "personality_scores": {{
        "analytical": 0–100,
        "creative": 0–100,
        "social": 0–100,
        "practical": 0–100,
        "investigative": 0–100
    }},
    "top_strength": "Most dominant quality (e.g. Problem-Solving)",
    "learning_style": "One of: Visual / Auditory / Kinesthetic / Reading",
    "potential_career_paths": [
        {{"path": "Career Title 1", "match_percentage": 90}},
        {{"path": "Career Title 2", "match_percentage": 75}},
        {{"path": "Career Title 3", "match_percentage": 65}}
    ],
    "recommended_next_steps": [
        "Action-oriented and practical step 1",
        "Action-oriented and practical step 2"
    ],
    "confidence_indicators": {{
        "decision_making": "High / Medium / Low",
        "self_awareness": "High / Medium / Low",
        "exploration_readiness": "High / Medium / Low"
    }},
    "interest_distribution": {{
        "STEM": 0–100,
        "Arts_Humanities": 0–100,
        "Business_Commerce": 0–100,
        "Social_Services": 0–100
    }},
    "your_natural_inclination": "Describe the graduate's strongest natural interests and patterns of motivation across skills/domains.",
    "careers_that_fit_you_well": [
        "Mention 1st career path that align with current interests and mindset, with quick reasoning.",
        "Mention 2nd career path that align with current interests and mindset, with quick reasoning.",
        "Or more"
    ],
    "your_degree_already_helps": "Show how their degree supports their career interests (directly or indirectly).",
    "what_you_can_do_next": [
        "Give practical and immediate next steps—skills, experience, or actions they can take today.",
        "Give practical and immediate next steps—skills, experience, or actions they can take today."
    ],
    "self_reflection": "Summarize any personal realizations, mindset shifts, or confidence built (optional, based on Q11/Q12).",
    "final_word": "A powerful and encouraging closing line to inspire confidence in stepping into the professional world."
    }}
    """

        prompt = PromptTemplate.from_template(prompt_template)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating LLM graduate profile (attempt {attempt + 1})")

                llm_response = self.llm.invoke(
                    prompt.format(
                        education_level=education_level,
                        formatted_responses=formatted_responses_str,
                        analyzed_patterns=analyzed_patterns
                    )
                )

                json_str = getattr(llm_response, 'content', str(llm_response)).strip()

                if not json_str:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                elif json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]

                json_str = json_str.strip()

                if not json_str.startswith('{'):
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx + 1]
                    else:
                        logger.warning(f"No valid JSON structure found on attempt {attempt + 1}")
                        continue


                profile_data = json.loads(json_str)

                required_fields = [
                    "profile_summary", "identified_keywords", "primary_orientation", "orientation_confidence",
                    "personality_scores", "top_strength", "learning_style", "potential_career_paths",
                    "recommended_next_steps", "confidence_indicators", "interest_distribution",
                    "your_natural_inclination", "careers_that_fit_you_well", "your_degree_already_helps",
                    "what_you_can_do_next", "self_reflection", "final_word"
                ]

                missing_fields = [field for field in required_fields if field not in profile_data]
                if missing_fields:
                    logger.warning(f"Missing fields: {missing_fields} on attempt {attempt + 1}")
                    continue

                logger.info("LLM graduate profile generated successfully")
                return profile_data

            except json.JSONDecodeError as e:
                logger.warning(f"JSON error on attempt {attempt + 1}: {e}")
                logger.warning(f"Problematic JSON string: {json_str[:500]}...")
                if attempt == max_retries - 1:
                    logger.error("Failed to generate valid JSON after all attempts")

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break

        logger.info("Using fallback graduate profile due to LLM failure")
        return {
            "profile_summary": "You have a strong foundational direction. You're capable of translating your interests into real career growth.",
            "identified_keywords": ["Focused", "Career-Ready", "Skill-Building", "Determined", "Insightful"],
            "primary_orientation": "Enterprising",
            "orientation_confidence": 80,
            "personality_scores": {
                "analytical": 65,
                "creative": 55,
                "social": 60,
                "practical": 70,
                "investigative": 50
            },
            "top_strength": "Self-Driven",
            "learning_style": "Reading",
            "potential_career_paths": [
                {"path": "Product Manager", "match_percentage": 90},
                {"path": "Operations Analyst", "match_percentage": 75},
                {"path": "Startup Associate", "match_percentage": 65}
            ],
            "recommended_next_steps": [
                "Join a project-based internship",
                "Take an advanced course aligned to your goal"
            ],
            "confidence_indicators": {
                "decision_making": "High",
                "self_awareness": "Medium",
                "exploration_readiness": "High"
            },
            "interest_distribution": {
                "STEM": 30,
                "Arts_Humanities": 20,
                "Business_Commerce": 35,
                "Social_Services": 15
            },
            "your_natural_inclination": "You gravitate toward building efficient systems and solving real-world problems.",
            "careers_that_fit_you_well": "Operations roles, startup teams, or scalable tech-based initiatives suit your strengths.",
            "your_degree_already_helps": "Your degree provides a solid platform — it aligns well with analytical and execution-based roles.",
            "what_you_can_do_next": "Identify one skill to master and get real-world exposure through freelancing or interning.",
            "self_reflection": "You’ve shown awareness of your direction, though a bit of clarity on long-term goals may help.",
            "final_word": "You’re at the starting line of real impact — stay bold, stay focused, and keep moving forward."
        }

# Example of how to use (optional, for testing within this file)
if __name__ == '__main__':
    # This is a placeholder for testing.
    # Ensure GROQ_API_KEY is set in your environment.
    # Ensure 'knowledge_base' directory exists with some .txt files in subdirectories.

    # Create dummy knowledge base for testing if it doesn't exist
    kb_dir = "knowledge_base_test_rag"
    if not os.path.exists(kb_dir):
        os.makedirs(os.path.join(kb_dir, "general"), exist_ok=True)
        with open(os.path.join(kb_dir, "general", "doc1.txt"), "w") as f:
            f.write(
                "This is a test document about general topics. AI is a fascinating field.")
        with open(os.path.join(kb_dir, "general", "doc2.txt"), "w") as f:
            f.write(
                "Another test document. It discusses machine learning and data science.")

    try:
        rag_system = RAGSystem(knowledge_dir=kb_dir)

        # Test query
        test_user_profile = {
            "education_level": "graduate",
            "profile_summary": "Interested in technology and AI.",
            "identified_keywords": ["AI", "machine learning", "software"]
        }

        question = "Tell me about AI and machine learning."
        print(f"\nQuerying RAG system with: '{question}'")

        response = rag_system.query(question, test_user_profile)

        print("\nResponse from RAG System:")
        print(f"Answer: {response.get('answer')}")

        if response.get('source_documents'):
            print("\nSource Documents:")
            for i, doc in enumerate(response['source_documents']):
                # Print snippet
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        else:
            print("No source documents retrieved.")

        print(f"\nToken Usage: {response.get('token_usage')}")

    except Exception as e:
        print(f"An error occurred during RAG system demo: {e}")
    finally:
        # Clean up dummy knowledge base
        if os.path.exists(kb_dir):
            shutil.rmtree(kb_dir)
            print(f"\nCleaned up test knowledge base: {kb_dir}")
