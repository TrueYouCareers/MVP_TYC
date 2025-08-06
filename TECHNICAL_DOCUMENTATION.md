# Technical Documentation - Career Guidance LLM Backend

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Groq LLM API  │────│  Vector Store   │
│  (Frontend)     │    │   (AI Core)     │    │   (Knowledge)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ RAG Controller  │
                    │ (groq_guidance) │
                    └─────────────────┘
```

### Component Interaction Flow

```
User Input → Streamlit → RAG Controller → Vector Store → LLM → Response
    ↑                                                            │
    └────────────────── User Interface ←─────────────────────────┘
```

## 🔧 Core Components Deep Dive

### 1. GroqCareerGuidanceRAG Class

#### Initialization Process

```python
def __init__(self, knowledge_dir="knowledge_base"):
    # 1. Initialize embeddings model
    self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Setup Groq LLM
    self.llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2048,
        groq_api_key=GROQ_API_KEY
    )

    # 3. Initialize tokenizer for context management
    self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # 4. Setup vector stores
    self._initialize_vector_stores()
```

#### Vector Store Architecture

```python
# Directory structure → Vector stores mapping
knowledge_base/
├── career_paths/     → vector_stores["career_paths"]
├── education_info/   → vector_stores["education_info"]
├── skills_guidance/  → vector_stores["skills_guidance"]
└── industry_trends/  → vector_stores["industry_trends"]
```

### 2. Profile Generation Pipeline

#### Step-by-Step Process

```python
def generate_llm_student_profile(education_level, responses, questions_data):
    # Step 1: Format responses for LLM
    formatted_responses = []
    for q_id, answer_key in responses.items():
        question_text = questions_data[q_id]["text"]
        answer_text = questions_data[q_id]["options"][answer_key]
        formatted_responses.append(f"Q: {question_text}\nA: {answer_text}")

    # Step 2: Create structured prompt
    prompt = create_profile_generation_prompt(formatted_responses, education_level)

    # Step 3: LLM processing with retry logic
    for attempt in range(3):
        try:
            response = self.llm.invoke(prompt)
            profile_data = json.loads(response.content)
            return validate_and_return(profile_data)
        except Exception as e:
            if attempt == 2:
                return fallback_profile()
            continue
```

#### Updated Profile Data Structure (Concise Version)

```python
{
    "profile_summary": str,          # 1-2 sentences max
    "identified_keywords": List[str], # 5 key traits/interests
    "primary_orientation": str,       # Single word personality type
    "orientation_confidence": int,    # 0-100 confidence score
    "personality_scores": {           # Visual data for charts
        "analytical": int,
        "creative": int,
        "social": int,
        "practical": int,
        "investigative": int
    },
    "top_strength": str,             # 1-2 words dominant trait
    "learning_style": str,           # Visual/Auditory/Kinesthetic/Reading
    "potential_career_paths": [      # 3 career options with match %
        {
            "path": str,
            "match_percentage": int
        }
    ],
    "recommended_next_steps": List[str],  # 2 actionable steps max
    "confidence_indicators": {       # Visual readiness assessment
        "decision_making": str,      # High/Medium/Low
        "self_awareness": str,
        "exploration_readiness": str
    },
    "interest_distribution": {       # Data for pie chart
        "STEM": int,
        "Arts_Humanities": int,
        "Business_Commerce": int,
        "Social_Services": int
    }
}
```

### 3. Visualization Components

#### Profile Display Features

- **Metrics Dashboard**: Key stats in card format
- **Personality Charts**: Bar charts and radar plots
- **Interest Distribution**: Pie charts showing area preferences
- **Career Match Indicators**: Progress bars for career fit
- **Readiness Assessment**: Color-coded confidence levels

#### Visual Elements Used

- Progress bars for career match percentages
- Bar charts for personality breakdown
- Pie charts for interest distribution
- Metric cards for key statistics
- Color-coded indicators for readiness levels

### 4. RAG Query Processing

#### Query Flow

```python
def query(question, user_profile, stream=False):
    # 1. Document Retrieval
    retriever = self.get_retriever()
    retrieved_docs = retriever.invoke(question)

    # 2. Context Optimization
    context_text = self._optimize_context(retrieved_docs, question, user_profile)

    # 3. Prompt Preparation
    prompt = self._prepare_context_and_prompt(question, user_profile)
    formatted_prompt = prompt.format(context=context_text, question=question)

    # 4. LLM Response Generation
    if stream:
        return {"answer_stream": self.llm.stream(formatted_prompt)}
    else:
        response = self.llm.invoke(formatted_prompt)
        return {"answer": response.content}
```

#### Context Optimization Strategy

```python
def _optimize_context(self, retrieved_docs, question, user_profile):
    context_parts = []
    current_tokens = 0
    max_context_tokens = MAX_CONTEXT_ALLOWED // 2  # Reserve space for response

    for doc in retrieved_docs[:6]:  # Top 6 documents
        doc_tokens = self._count_tokens(doc.page_content)

        if current_tokens + doc_tokens > max_context_tokens:
            # Truncate document to fit
            remaining_tokens = max_context_tokens - current_tokens
            if remaining_tokens > 100:
                truncated_text = self._truncate_to_tokens(doc.page_content, remaining_tokens)
                context_parts.append(truncated_text)
            break
        else:
            context_parts.append(doc.page_content)
            current_tokens += doc_tokens

    return "\n\n---\n\n".join(context_parts)
```

## 🎯 Prompt Engineering

### Profile Generation Prompt Structure

```python
PROMPT_TEMPLATE = """
You are Dr. Priya Sharma, an expert career psychologist.

CRITICAL: Respond with ONLY valid JSON.

TASK: Create a personality profile for a {education_level} student.

STUDENT RESPONSES:
{formatted_responses}

REQUIRED OUTPUT FORMAT:
{
  "profile_summary": "...",
  "identified_keywords": [...],
  "primary_orientation": "...",
  "potential_career_paths": [...],
  "recommended_next_steps": [...],
  "personality_insights": {...}
}
"""
```

### Counseling Prompt Structure

```python
COUNSELING_TEMPLATE = """
🎓 ROLE: Expert AI career counselor with 15+ years experience

STUDENT PROFILE:
📚 Education Level: {education_level}
🎯 Profile Summary: {profile_summary}
💡 Key Interests: {identified_keywords}
🧭 Primary Orientation: {primary_orientation}

💬 COMMUNICATION STYLE:
- Warm, encouraging, genuinely interested
- Conversational tone like a wise mentor
- Keep responses under 200 words
- Ask thoughtful follow-up questions

📖 RETRIEVED KNOWLEDGE:
{context}

💭 STUDENT'S QUESTION: {question}

🗣️ YOUR RESPONSE:
Structure: [Acknowledgment] → [2-3 key points] → [Follow-up questions]
"""
```

## 💾 Data Management

### Session State Management

```python
# Streamlit session state structure
st.session_state = {
    'page': 'education_level|questionnaire|results|chatbot',
    'education_level': '10th|12th|graduate',
    'responses': {q_id: answer_key, ...},
    'user_profile': {...},
    'messages': [{'role': 'user|assistant', 'content': '...'}, ...],
    'rag': GroqCareerGuidanceRAG(),
    'filename': 'user_profile_timestamp.json'
}
```

### File Storage Format

```python
# user_profile_YYYYMMDD_HHMMSS.json
{
    "education_level": "12th",
    "timestamp": "20241201_143025",
    "raw_responses": {
        "q1": "a",
        "q2": "b",
        # ... all question responses
    },
    "llm_profile": {
        # Generated AI profile
    }
}
```

## 🔄 Application Flow Diagrams

### Assessment Flow

```
Start
  ↓
Education Level Selection
  ↓
Load Dynamic Questions
  ↓
Collect User Responses
  ↓
Initialize RAG System
  ↓
Generate LLM Profile
  ↓
Validate Profile Data
  ↓
Save to JSON File
  ↓
Display Results
  ↓
End
```

### Chat Flow

```
Start
  ↓
Upload Profile JSON
  ↓
Parse and Validate Profile
  ↓
Initialize Chat Interface
  ↓
User Asks Question
  ↓
Retrieve Relevant Context
  ↓
Generate Contextual Prompt
  ↓
Stream LLM Response
  ↓
Display Response + Sources
  ↓
Continue Chat Loop
```

## 🚀 Performance Optimizations

### 1. Token Management

```python
# Context window optimization
MAX_CONTEXT_TOKENS = 128000
RESERVED_TOKENS = 3000
MAX_CONTEXT_ALLOWED = MAX_CONTEXT_TOKENS - RESERVED_TOKENS

# Dynamic context truncation
def _truncate_to_tokens(text, max_tokens):
    max_chars = max_tokens * 4  # Rough estimation
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated + "..."
```

### 2. Caching Strategies

```python
# Streamlit resource caching
@st.cache_resource
def initialize_rag_system():
    rag = GroqCareerGuidanceRAG()
    rag.setup_knowledge_base(SAMPLE_KNOWLEDGE)
    return rag

# Vector store persistence
# FAISS indexes are automatically saved and loaded
```

### 3. Streaming Responses

```python
# Real-time response streaming
def stream_response(prompt):
    full_response = ""
    message_placeholder = st.empty()

    for chunk in self.llm.stream(prompt):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        full_response += content
        message_placeholder.markdown(full_response)

    return full_response
```

## 🛡️ Error Handling

### LLM Error Handling

```python
def generate_llm_student_profile(self, ...):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm_response = self.llm.invoke(prompt)
            profile_data = json.loads(llm_response.content)

            # Validate required fields
            required_fields = ["profile_summary", "identified_keywords", ...]
            missing_fields = [f for f in required_fields if f not in profile_data]
            if missing_fields:
                continue

            return profile_data

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return fallback_profile()
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return fallback_profile()
```

### Network Error Handling

```python
# Groq client configuration with retries
self.llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    max_retries=3,
    request_timeout=60,
    groq_api_key=GROQ_API_KEY
)
```

## 📊 Monitoring and Logging

### Logging Configuration

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key logging points
logger.info("Embeddings initialized successfully")
logger.info(f"Retrieved {len(retrieved_docs)} documents")
logger.warning("JSON parsing error - retrying")
logger.error("Failed to generate valid JSON after all attempts")
```

### Performance Metrics

```python
# Token usage tracking
return {
    "answer": response,
    "token_usage": {
        "template": template_tokens,
        "context": context_tokens,
        "total_input": total_tokens
    }
}
```

## 🧪 Testing Framework

### Unit Tests

```python
# Test profile generation
def test_profile_generation():
    rag = GroqCareerGuidanceRAG()
    sample_responses = {"q1": "a", "q2": "b"}
    sample_questions = {...}

    profile = rag.generate_llm_student_profile("12th", sample_responses, sample_questions)

    assert "profile_summary" in profile
    assert len(profile["identified_keywords"]) > 0
    assert "potential_career_paths" in profile

# Test RAG query
def test_rag_query():
    rag = GroqCareerGuidanceRAG()
    user_profile = {...}

    result = rag.query("What should I study after 12th?", user_profile)

    assert "answer" in result
    assert len(result["source_documents"]) > 0
```

### Integration Tests

```python
# Test end-to-end flow
def test_complete_flow():
    # 1. Generate profile
    profile = generate_test_profile()

    # 2. Save to file
    filename = save_profile_data(...)

    # 3. Load and query
    loaded_profile = load_user_profile(filename)
    result = rag.query("Test question", loaded_profile)

    assert result["answer"] is not None
```

This technical documentation provides a comprehensive understanding of the system architecture, component interactions, and implementation details for developers working on or integrating with the Career Guidance LLM Backend.
