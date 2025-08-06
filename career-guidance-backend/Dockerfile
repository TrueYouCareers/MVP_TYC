# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry settings (if you decide to use Poetry for dependency management later)
    # POETRY_VERSION=1.7.1 \
    # POETRY_HOME="/opt/poetry" \
    # POETRY_VIRTUALENVS_CREATE=false \
    # PATH="$POETRY_HOME/bin:$PATH" \
    # FastAPI/Uvicorn settings
    APP_MODULE="app.main:app" \
    HOST="0.0.0.0" \
    PORT="8000"

# Create and set the working directory
WORKDIR /app

# Install system dependencies (if any are needed beyond base Python)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
# This includes your 'app' directory, 'alembic.ini', 'knowledge_base', etc.
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
# The --reload flag is useful for development. Remove it for production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]