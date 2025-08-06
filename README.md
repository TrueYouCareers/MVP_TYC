# Career Guidance LLM Backend

A comprehensive AI-powered career guidance system that provides personalized career counseling for Indian students using Groq LLM, RAG (Retrieval Augmented Generation), and Streamlit. The system analyzes student responses to provide tailored career recommendations based on their education level, interests, and personality traits.

## 🎯 Project Overview

This project addresses the critical need for personalized career guidance in India, where millions of students struggle with career decisions. Our AI-powered system:

- Provides personalized career assessments for 10th, 12th, and graduate students
- Uses advanced LLM technology to generate concise, visual personality profiles
- Offers contextual career guidance through a RAG-based knowledge system
- Delivers recommendations tailored to the Indian education system and job market
- Features beautiful data visualizations for better profile understanding

## 🏗️ Project Architecture

```
Career Guidance LLM Backend
├── Assessment Layer (questionnaire.py)
├── AI Analysis Layer (groq_guidance.py) - Enhanced with visual data
├── Knowledge Base (supporting_context_docs/)
├── User Interface (integrated_app.py, chatbot_ui.py) - Visual enhancements
├── Docker Infrastructure (Dockerfile, docker-compose.yml)
└── Supporting Files (llmflow.py, requirements.txt)
```

## 📁 File Structure and Documentation

### Core Application Files

#### 1. `integrated_app.py` - Main Streamlit Application

**Purpose**: Multi-page Streamlit application serving as the primary user interface

**Key Features**:

- Assessment questionnaire interface
- AI-powered career counselor chat
- User profile management and visualization
- Results display and analysis

**Flow**:

1. User selects education level (10th/12th/Graduate)
2. Completes assessment questionnaire
3. AI generates personality profile
4. User can chat with AI counselor for guidance
5. System provides personalized recommendations

#### 2. `questionnaire.py` - Assessment Module

**Purpose**: Standalone assessment interface for collecting user responses

**Components**:

- Education level selection
- Dynamic question sets based on education level
- Response collection and validation
- AI profile generation integration
- Results saving and display

**Question Categories by Education Level**:

- **10th Grade**: Stream selection, natural inclinations, career interests
- **12th Grade**: Subject choices, career exploration, future aspirations
- **Graduate**: IT career transition, skill assessment, professional goals

#### 3. `groq_guidance.py` - Core AI/RAG System

**Purpose**: Heart of the AI system - handles LLM interactions and knowledge retrieval

**Key Classes and Methods**:

```python
class GroqCareerGuidanceRAG:
    def __init__(knowledge_dir="knowledge_base")
    def setup_knowledge_base(data: Dict[str, List[str]])
    def generate_llm_student_profile(education_level, responses, questions_data)  # Enhanced for concise output
    def query(question, user_profile, stream=False)
    def _prepare_context_and_prompt(question, user_profile)
```

**Core Capabilities**:

- **Vector Store Management**: Creates FAISS vector stores for knowledge retrieval
- **Concise LLM Profile Generation**: Analyzes responses to create brief, visual-friendly profiles
- **Contextual Querying**: Provides personalized answers using RAG
- **Token Optimization**: Manages context windows for optimal performance

#### 4. `chatbot_ui.py` - Interactive Chat Interface

**Purpose**: Dedicated chat interface for career counseling conversations

**Features**:

- File upload for user profiles
- Real-time streaming responses
- Source document display
- Suggested questions
- Chat history management

### Knowledge Base and Context

#### 5. `supporting_context_docs/` - Knowledge Repository

**Contents**:

- `Knowledge Base.txt`: Career counseling theories and frameworks
- `KnowledgeBase(II-V).txt`: Detailed career guidance for different education levels
- Career path recommendations for BA, B.Com, B.Sc, BBA graduates
- Indian education system context and job market insights

**Knowledge Categories**:

- **Stream Selection**: Science, Commerce, Arts guidance for 10th students
- **Career Exploration**: Detailed paths for 12th students
- **Professional Transition**: IT and corporate careers for graduates
- **Skills Development**: Technical and non-technical skill tracks

### Supporting Files

#### 6. `llmflow.py` - Legacy/Alternative Implementation

**Purpose**: Contains alternative implementation approaches (currently commented out)

- Google Gemini integration templates
- Different RAG architectures
- Sample knowledge base structures

### Infrastructure and Deployment

#### 7. Docker Configuration

- `Dockerfile`: Multi-stage Docker build with security best practices
- `docker-compose.yml`: Container orchestration
- `.dockerignore`: Optimized build context
- `requirements.txt`: Python dependencies

#### 8. Environment and Security

- `.env.example`: Environment variable templates
- API key management through environment variables
- Non-root user execution in containers

## 🔄 Application Flow

### 1. Assessment Flow

```
User Selects Education Level →
Dynamic Questions Load →
User Completes Assessment →
Raw Responses Collected →
AI Generates Personality Profile →
Profile Saved to JSON →
Results Displayed
```

### 2. Counseling Flow

```
User Uploads Profile JSON →
Profile Parsed and Analyzed →
Chat Interface Activated →
User Asks Questions →
RAG System Retrieves Context →
LLM Generates Personalized Response →
Response Streamed to User
```

### 3. AI Processing Pipeline

```
Raw Assessment Responses →
Structured Prompt Creation →
LLM Analysis (Groq) →
JSON Profile Generation →
Validation and Error Handling →
Profile Storage
```

## 🧠 AI System Architecture

### LLM Integration (Groq)

- **Model**: llama-3.3-70b-versatile
- **Temperature**: 0.7 for natural, empathetic responses
- **Token Management**: Optimized context windows
- **Error Handling**: Retry logic and fallback mechanisms

### RAG (Retrieval Augmented Generation)

- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS for efficient similarity search
- **Chunking**: Optimized text splitting (800 chars, 150 overlap)
- **Retrieval**: Ensemble retriever with score thresholds

### Prompt Engineering

- **Role Definition**: Expert AI career counselor persona
- **Context Awareness**: Education level and profile-specific guidance
- **Response Structure**: Acknowledgment → Advice → Follow-up questions
- **Indian Context**: Tailored for Indian education system and job market

## 📊 Data Models

### Updated User Profile Structure (Concise & Visual)

```json
{
  "education_level": "10th|12th|graduate",
  "timestamp": "YYYYMMDD_HHMMSS",
  "raw_responses": {"q1": "a", "q2": "b", ...},
  "llm_profile": {
    "profile_summary": "1-2 sentence natural inclination summary",
    "identified_keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
    "primary_orientation": "Analytical|Creative|Social|Practical|Investigative|Enterprising",
    "orientation_confidence": 85,
    "personality_scores": {
      "analytical": 75,
      "creative": 60,
      "social": 45,
      "practical": 80,
      "investigative": 70
    },
    "top_strength": "Most dominant trait",
    "learning_style": "Visual|Auditory|Kinesthetic|Reading",
    "potential_career_paths": [
      {"path": "Career Name", "match_percentage": 90}
    ],
    "recommended_next_steps": ["Step1", "Step2"],
    "confidence_indicators": {
      "decision_making": "High|Medium|Low",
      "self_awareness": "High|Medium|Low",
      "exploration_readiness": "High|Medium|Low"
    },
    "interest_distribution": {
      "STEM": 40,
      "Arts_Humanities": 20,
      "Business_Commerce": 25,
      "Social_Services": 15
    }
  }
}
```

### API Response Structure

```json
{
  "answer": "AI generated response",
  "source_documents": [{"page_content": "...", "metadata": {...}}],
  "token_usage": {
    "template": 1000,
    "context": 2000,
    "total_input": 3000
  }
}
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Groq API Key

### Updated Dependencies

```bash
# Additional visualization library
pip install plotly>=5.0.0
```

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd CareerGuidanceLLM-backend

# Setup environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Run with Docker (Recommended)
chmod +x docker-build.sh docker-run.sh
./docker-build.sh
./docker-run.sh

# Access application with enhanced visuals
open http://localhost:8501
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Run applications
streamlit run integrated_app.py  # Main app
streamlit run questionnaire.py  # Assessment only
streamlit run chatbot_ui.py     # Chat only
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Required - Your Groq API key
- `STREAMLIT_SERVER_PORT`: Default 8501
- `DEFAULT_MODEL`: LLM model name
- `EMBEDDING_MODEL`: Embedding model for RAG

### Model Configuration

- **LLM Settings**: Temperature, max tokens, retry logic
- **RAG Settings**: Chunk size, overlap, retrieval parameters
- **UI Settings**: Streaming, caching, session management

## 📈 Performance Optimization

### Token Management

- Context window optimization (128K tokens)
- Intelligent document truncation
- Response length limits

### Caching Strategies

- Streamlit resource caching for RAG system
- Vector store persistence
- Session state management

### Error Handling

- Graceful LLM failures with fallback profiles
- JSON parsing validation
- Network retry mechanisms

## 🛡️ Security Features

### Docker Security

- Non-root user execution
- Multi-stage builds
- Minimal attack surface
- Health checks

### Data Security

- Environment variable for secrets
- Input validation and sanitization
- Safe file handling
- No hardcoded credentials

## 🧪 Testing and Validation

### Profile Generation Testing

```python
# Test LLM profile generation
rag = GroqCareerGuidanceRAG()
profile = rag.generate_llm_student_profile(
    education_level="12th",
    responses=sample_responses,
    questions_data=sample_questions
)
```

### RAG System Testing

```python
# Test knowledge retrieval
result = rag.query(
    "What career options do I have after B.Com?",
    user_profile=sample_profile
)
```

## 🔮 Future Enhancements

### Planned Features

- Multi-language support (Hindi, regional languages)
- Voice-based assessments
- Career path visualization
- Institution recommendations
- Skill gap analysis
- Progress tracking

### Technical Improvements

- Advanced RAG with re-ranking
- Fine-tuned models for Indian context
- Mobile app development
- API-first architecture
- Analytics dashboard

### Planned Visual Features

- 3D personality radar charts
- Interactive career exploration maps
- Animated progress indicators
- Dark/light theme options
- Export profile as infographic

### Enhanced Analytics

- Comparison with peer profiles
- Industry trend overlays
- Skill gap visualizations
- Career path flowcharts

## 🤝 Contributing

### Development Workflow

1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests and documentation
5. Submit pull request

### Code Standards

- Python PEP 8 compliance
- Comprehensive docstrings
- Type hints where applicable
- Error handling best practices

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

For technical support or questions:

- Create an issue in the repository
- Review documentation and examples
- Check troubleshooting guide

---

**Built with ❤️ for empowering student career decisions in India through beautiful, concise insights**
