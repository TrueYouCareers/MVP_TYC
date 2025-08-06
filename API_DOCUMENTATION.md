# API Documentation - Career Guidance LLM Backend

## 🚀 API Overview

The Career Guidance LLM Backend provides both programmatic access and web interfaces for AI-powered career counseling. The system is built around a core RAG (Retrieval Augmented Generation) API with Streamlit web interfaces.

## 🏗️ API Architecture

### Core API Class: GroqCareerGuidanceRAG

The main API is exposed through the `GroqCareerGuidanceRAG` class, which provides methods for:

- Student profile generation
- Knowledge-based query processing
- Career guidance recommendations

### Web Interfaces

- **Integrated App**: Multi-page application (`integrated_app.py`)
- **Assessment Only**: Questionnaire interface (`questionnaire.py`)
- **Chat Only**: Counseling interface (`chatbot_ui.py`)

## 📋 API Reference

### Class: GroqCareerGuidanceRAG

#### Constructor

```python
GroqCareerGuidanceRAG(knowledge_dir="knowledge_base")
```

**Parameters:**

- `knowledge_dir` (str): Directory containing knowledge base files

**Returns:** Initialized RAG system instance

**Example:**

```python
from groq_guidance import GroqCareerGuidanceRAG

# Initialize with default knowledge base
rag = GroqCareerGuidanceRAG()

# Initialize with custom knowledge directory
rag = GroqCareerGuidanceRAG(knowledge_dir="custom_knowledge")
```

---

#### Method: setup_knowledge_base()

```python
setup_knowledge_base(data: Dict[str, List[str]]) -> None
```

**Purpose:** Initialize knowledge base from provided data

**Parameters:**

- `data` (Dict[str, List[str]]): Knowledge organized by categories

**Example:**

```python
knowledge_data = {
    "career_paths": [
        "Software engineering requires strong programming skills...",
        "Medical careers demand scientific aptitude..."
    ],
    "education_info": [
        "Engineering colleges in India include IITs, NITs...",
        "Medical entrance exams include NEET, AIIMS..."
    ]
}

rag.setup_knowledge_base(knowledge_data)
```

---

#### Method: generate_llm_student_profile()

```python
generate_llm_student_profile(
    education_level: str,
    responses: Dict[str, str],
    questions_data: Dict[str, Any]
) -> Dict[str, Any]
```

**Purpose:** Generate AI-powered student personality and career profile

**Parameters:**

- `education_level` (str): Student's education level ("10th", "12th", "graduate")
- `responses` (Dict[str, str]): Question ID to answer key mapping
- `questions_data` (Dict[str, Any]): Complete question definitions

**Returns:** Generated profile dictionary

**Response Schema:**

```python
{
    "profile_summary": str,
    "identified_keywords": List[str],
    "primary_orientation": str,
    "potential_career_paths": [
        {
            "path": str,
            "reasoning": str
        }
    ],
    "recommended_next_steps": List[str],
    "personality_insights": {
        "confidence_level_summary": str,
        "decision_making_drivers": str
    }
}
```

**Example:**

```python
# Sample responses for 12th grade student
responses = {
    "q1": "a",  # Science Stream
    "q2": "a",  # Genuine Interest
    "q3": "a"   # Very confident
}

# Sample question data
questions_data = {
    "q1": {
        "text": "Which stream did you choose?",
        "options": {
            "a": "Science Stream",
            "b": "Arts Stream",
            "c": "Commerce Stream"
        }
    }
}

profile = rag.generate_llm_student_profile(
    education_level="12th",
    responses=responses,
    questions_data=questions_data
)

print(profile["profile_summary"])
# Output: "You show strong analytical thinking and scientific curiosity..."
```

---

#### Method: query()

```python
query(
    question: str,
    user_profile: Dict[str, Any],
    stream: bool = False
) -> Dict[str, Any]
```

**Purpose:** Process user questions with personalized context

**Parameters:**

- `question` (str): User's question
- `user_profile` (Dict[str, Any]): Student profile for personalization
- `stream` (bool): Whether to return streaming response

**Returns:** Response dictionary with answer and metadata

**Response Schema:**

```python
{
    "answer": str,  # AI-generated response
    "source_documents": List[Document],  # Retrieved knowledge
    "token_usage": {
        "template": int,
        "context": int,
        "total_input": int
    }
}

# For streaming responses:
{
    "answer_stream": Iterator,  # Streaming response generator
    "source_documents": List[Document],
    "token_usage": Dict[str, int]
}
```

**Example:**

```python
# Standard query
result = rag.query(
    question="What should I study after 12th for engineering?",
    user_profile=student_profile
)

print(result["answer"])
# Output: "Based on your strong mathematical aptitude and interest in technology..."

# Streaming query
stream_result = rag.query(
    question="Tell me about career options in data science",
    user_profile=student_profile,
    stream=True
)

for chunk in stream_result["answer_stream"]:
    print(chunk.content, end="")
```

---

#### Method: load_user_responses()

```python
load_user_responses(json_file: str) -> Dict[str, Any]
```

**Purpose:** Load user assessment data from JSON file

**Parameters:**

- `json_file` (str): Path to JSON file containing user responses

**Returns:** Complete user data including responses and profile

**Example:**

```python
user_data = rag.load_user_responses("user_profile_20241201_143025.json")

print(user_data["education_level"])  # "12th"
print(user_data["llm_profile"]["profile_summary"])
```

---

#### Method: get_retriever()

```python
get_retriever(category: Optional[str] = None) -> BaseRetriever
```

**Purpose:** Get document retriever for specific knowledge category

**Parameters:**

- `category` (Optional[str]): Knowledge category or None for all categories

**Returns:** Configured retriever instance

**Example:**

```python
# Get retriever for all categories
all_retriever = rag.get_retriever()

# Get retriever for specific category
career_retriever = rag.get_retriever("career_paths")

# Use retriever directly
docs = career_retriever.invoke("software engineering careers")
```

## 🌐 Web Interface APIs

### Streamlit Application Endpoints

#### Integrated Application (`integrated_app.py`)

**URL:** `http://localhost:8501`

**Pages:**

- `/` - Education level selection
- `/assessment` - Dynamic questionnaire
- `/results` - Profile results display
- `/chatbot` - AI counselor chat

#### Assessment Application (`questionnaire.py`)

**URL:** `http://localhost:8502` (if run separately)

**Features:**

- Education level selection
- Assessment questionnaire
- Profile generation
- Results download

#### Chat Application (`chatbot_ui.py`)

**URL:** `http://localhost:8503` (if run separately)

**Features:**

- Profile upload
- Real-time chat
- Source document display
- Suggested questions

## 📊 Data Formats

### Assessment Response Format

```python
{
    "education_level": "10th|12th|graduate",
    "timestamp": "YYYYMMDD_HHMMSS",
    "raw_responses": {
        "q1": "a",
        "q2": "b",
        "q3": "c"
        # ... all question responses
    },
    "llm_profile": {
        # Generated AI profile
    }
}
```

### Question Definition Format

```python
{
    "q1": {
        "text": "What do you enjoy doing most?",
        "options": {
            "a": "Solving puzzles and logical problems",
            "b": "Reading and researching",
            "c": "Creating and designing",
            "d": "Leading and organizing"
        }
    }
}
```

### Career Path Format

```python
{
    "path": "Software Engineer",
    "reasoning": "Your logical thinking and interest in technology align well with programming careers"
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key

# Optional
STREAMLIT_SERVER_PORT=8501
DEFAULT_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Model Configuration

```python
# LLM Settings
{
    "model_name": "llama-3.3-70b-versatile",
    "temperature": 0.7,
    "max_tokens": 2048,
    "max_retries": 3
}

# RAG Settings
{
    "chunk_size": 800,
    "chunk_overlap": 150,
    "similarity_threshold": 0.4,
    "max_docs": 6
}
```

## 📝 Usage Examples

### Complete Workflow Example

```python
from groq_guidance import GroqCareerGuidanceRAG
import json

# 1. Initialize system
rag = GroqCareerGuidanceRAG()

# 2. Generate student profile
responses = {"q1": "a", "q2": "b", "q3": "a"}
questions = load_questions_for_level("12th")

profile = rag.generate_llm_student_profile(
    education_level="12th",
    responses=responses,
    questions_data=questions
)

# 3. Save profile
with open("student_profile.json", "w") as f:
    json.dump({
        "education_level": "12th",
        "timestamp": "20241201_143025",
        "raw_responses": responses,
        "llm_profile": profile
    }, f)

# 4. Query for guidance
result = rag.query(
    question="What engineering branches should I consider?",
    user_profile=profile
)

print("AI Counselor:", result["answer"])

# 5. Show sources
for i, doc in enumerate(result["source_documents"]):
    print(f"Source {i+1}: {doc.page_content[:100]}...")
```

### Batch Processing Example

```python
# Process multiple students
students = [
    {"responses": {...}, "education_level": "12th"},
    {"responses": {...}, "education_level": "graduate"}
]

rag = GroqCareerGuidanceRAG()

for student in students:
    profile = rag.generate_llm_student_profile(
        student["education_level"],
        student["responses"],
        questions_data
    )

    # Save or process profile
    print(f"Generated profile for {student['education_level']} student")
```

### Custom Knowledge Base Example

```python
# Create custom knowledge base
custom_knowledge = {
    "tech_careers": [
        "Cloud computing roles are growing rapidly...",
        "AI/ML engineers need strong math background..."
    ],
    "non_tech_careers": [
        "Digital marketing combines creativity with analytics...",
        "Product management requires strategic thinking..."
    ]
}

rag = GroqCareerGuidanceRAG()
rag.setup_knowledge_base(custom_knowledge)

# Query with custom knowledge
result = rag.query(
    "What are emerging career options in technology?",
    user_profile
)
```

## 🚨 Error Handling

### Common Error Responses

```python
# Profile generation failure
{
    "error": "Profile generation failed",
    "details": "JSON parsing error after 3 attempts"
}

# API key missing
ValueError: "GROQ_API_KEY environment variable is required"

# Knowledge base not found
FileNotFoundError: "Knowledge directory 'knowledge_base' not found"

# Invalid education level
ValueError: "Education level must be '10th', '12th', or 'graduate'"
```

### Retry Logic

```python
# Automatic retries for LLM calls
max_retries = 3
for attempt in range(max_retries):
    try:
        response = llm.invoke(prompt)
        return process_response(response)
    except Exception as e:
        if attempt == max_retries - 1:
            return fallback_response()
        continue
```

## 🔒 Security Considerations

### API Key Management

- Store API keys in environment variables
- Never commit keys to version control
- Use different keys for development/production

### Input Validation

- Validate education levels
- Sanitize user inputs
- Limit response lengths

### Rate Limiting

- Implement request throttling for production use
- Monitor API usage and costs
- Set appropriate timeout values

This API documentation provides comprehensive guidance for developers integrating with or extending the Career Guidance LLM Backend system.
