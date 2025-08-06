from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

# Default to local Dockerized instance if MONGODB_URI is not set in .env
# For local Docker: "mongodb://mongo:27017/LLM_db" (if 'mongo' is the service name)
# For Atlas: your Atlas URI
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/LLM_db")

client = None
db = None


def connect_to_mongo():
    global client, db
    if MONGODB_URI is None:
        raise ValueError("MONGODB_URI environment variable not set.")

    # Ensure ServerApi is only used for Atlas URIs if needed, or remove if connecting to local non-Atlas
    if "mongodb+srv" in MONGODB_URI:
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    else:
        client = MongoClient(MONGODB_URI)

    # The database name can be part of the MONGODB_URI or specified here.
    # If it's in the URI (e.g., mongodb://host/dbname), PyMongo uses it.
    # Otherwise, you can specify it:
    db_name = MONGODB_URI.split(
        '/')[-1].split('?')[0] if '/' in MONGODB_URI else "LLM_db"
    db = client["LLMCluster"]

    try:
        # Check connection
        client.admin.command('ping')
        print("MongoDB connected. Available collections:",
              db.list_collection_names())
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        # Potentially raise an error or handle appropriately
        return None  # Or raise
    return db


def close_mongo_connection():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")
