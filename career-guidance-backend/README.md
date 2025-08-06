# Career Guidance Backend

This project implements a career guidance system using FastAPI and Langchain agents. It provides personalized career advice based on user profiles and queries, leveraging a knowledge base and a retrieval-augmented generation (RAG) system.

## Project Structure

```
career-guidance-backend/
├── app/
│   ├── api/                     # API layer
│   │   ├── dependencies.py      # Dependency functions for routes
│   │   ├── errors.py            # Custom error handling
│   │   └── routes/              # API routes
│   ├── core/                    # Core application logic
│   ├── db/                      # Database models and session management
│   ├── models/                  # Data models
│   ├── schemas/                 # Pydantic schemas for data validation
│   ├── services/                # Business logic services
│   ├── llm/                     # Language model integration
│   ├── utils/                   # Utility functions
│   └── main.py                  # Entry point for the application
├── knowledge_base/              # Directory containing knowledge base files
├── tests/                       # Test suite for the application
├── alembic/                     # Database migration scripts
├── alembic.ini                  # Alembic configuration file
├── pyproject.toml               # Project dependencies and settings
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## User Flow

The career guidance platform follows this general user flow:

1.  **User Registration/Login**:

    - New users sign up with their email, password, education level, full name, and contact. This data is stored in MySQL.
    - Existing users log in using their credentials.
    - Authentication is handled via JWT (JSON Web Tokens).

2.  **Questionnaire Submission**:

    - After login/registration, the user is directed to a questionnaire tailored to their education level (10th, 12th, Graduate).
    - **Questions are defined and managed in the React frontend** with different question sets for each education level.
    - Upon submission:
      - The frontend sends the `user_id`, `education_level`, `raw_responses` (answers), and optionally `questions_data` (full question context) to the backend `POST /profile/questionnaire` endpoint.
      - The backend uses the RAG system (`RAGSystem.generate_llm_student_profile`) to generate an AI-driven profile analysis based on the responses.
      - The backend `ProfileService` saves the comprehensive data (raw responses, LLM profile, timestamp, user ID, education level) into a MongoDB collection (`user_questionnaires`).

3.  **Initial Profile Display**:

    - The frontend displays the AI-generated profile analysis to the user immediately after questionnaire submission. This includes a profile summary, key interests/skills, primary orientation, potential career paths, and recommended next steps.

4.  **Interactive Chat Guidance**:

    - The user can proceed to a chat interface to ask specific questions.
    - When the user sends a query:
      - The frontend sends a request to `POST /guidance/chat` with the user's question, user_id, and optionally a session_id.
      - The backend chat endpoint retrieves the user's saved questionnaire data (raw responses and LLM-generated profile) from MongoDB using the `ProfileService`.
      - This user-specific context, the ongoing chat history, and relevant documents retrieved from the knowledge base (vector store), are fed into the RAG system (`RAGSystem.query`).
      - The LLM generates a personalized answer using the retrieved context and user profile, which is returned to the user.
      - Chat history is saved in MongoDB (`chat_sessions` collection) for context in future interactions.
      - If no session_id is provided, a new chat session is automatically created.

5.  **Report Generation**:
    - Users can request a comprehensive career guidance report via the `GET /report/generate/{user_id}` endpoint.
    - The backend service:
      - Fetches the user's LLM-generated profile and their complete chat history from MongoDB.
      - Uses the LLM to intelligently fill a predefined template (specific to the user's education level) using the profile and chat data.
      - Converts the filled report string into a PDF document.
      - Returns the PDF report to the user for download/display.

## Database Migrations (MySQL with Alembic)

This project uses Alembic to manage MySQL database schema migrations. When you make changes to the SQLAlchemy models (located in `app/models/`), you need to generate and apply migrations to update your database schema. Migrations are **not** applied automatically.

The Alembic environment (`alembic/env.py`) is configured to detect changes in your models.

### Generating Migrations

After changing your SQLAlchemy models (e.g., adding a new table or column):

1.  Open your terminal in the project root directory.
2.  Run the following command to let Alembic autogenerate a migration script:
    ```bash
    alembic revision -m "your_descriptive_message_here" --autogenerate
    ```
    Replace `"your_descriptive_message_here"` with a short description of the changes (e.g., `"add_user_phone_number_column"`).
3.  A new migration file will be created in the `alembic/versions/` directory. **Always review this generated script** to ensure it accurately reflects your intended schema changes.

### Applying Migrations

To apply the generated (and reviewed) migrations to your database:

1.  Ensure your database server is running and accessible.
2.  Run the following command:
    ```bash
    alembic upgrade head
    ```
    This command applies all pending migrations up to the latest version (`head`).

### Downgrading Migrations (Use with Caution)

You can also revert migrations:

- `alembic downgrade -1`: Reverts the last applied migration.
- `alembic downgrade base`: Reverts all migrations.

### Initial Setup

If you are setting up the database for the first time after cloning the repository and models already exist:

1. Ensure your `.env` file has the correct `MYSQL_` connection details.
2. Create the database specified in `MYSQL_DATABASE` if it doesn't exist.
3. Run `alembic upgrade head` to apply all existing migrations and create the schema.

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd career-guidance-backend
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Set up the environment variables by copying `.env.example` to `.env` and updating the values as needed. Ensure your MySQL and MongoDB connection details are correctly configured.

5. **Database Migrations (MySQL)**:
   This project uses SQLAlchemy for ORM with MySQL. For managing database schema changes, Alembic is configured.
   - To initialize the database with Alembic (after setting up `alembic.ini` and `env.py`):
     ```bash
     alembic revision -m "create initial tables"
     alembic upgrade head
     ```
   - For subsequent model changes:
     ```bash
     alembic revision -m "describe your changes"
     alembic upgrade head
     ```
     (Note: The `Base.metadata.create_all(bind=engine)` in `main.py` can create tables for development but Alembic is preferred for production and managing schema evolution.)

## Running the Application

To start the FastAPI application, run:

```
uvicorn app.main:app --reload
```

You can access the API documentation at `http://127.0.0.1:8000/docs`.

## Testing

To run the tests, use:

```
pytest
```

## API Endpoints

### Chat Endpoints

- `POST /guidance/chat` - Send a message to the career guidance system
- `GET /guidance/chat/history/{user_id}/{session_id}` - Get chat session history
- `GET /guidance/chat/sessions/{user_id}` - Get all chat sessions for a user

### Profile Endpoints

- `POST /profile/questionnaire` - Submit questionnaire responses and generate LLM profile
- `GET /profile/questionnaire/{user_id}` - Get user's questionnaire data and profile
- `PUT /profile/questionnaire/{user_id}/profile` - Update user's LLM profile

### Report Endpoints

- `GET /report/generate/{user_id}` - Generate a comprehensive career guidance report as a PDF document.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
