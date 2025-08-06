from app.api.routes import auth, profile, guidance, report, admin, question
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.mongo import connect_to_mongo, close_mongo_connection
from app.db.session import engine 
from app.models import Base 
from app.db.connections import set_db_connections, clear_db_connections
from app.api.routes.report import router as report_router
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


mongodb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables if they don't exist (for development)
    # For production, use Alembic migrations.
    Base.metadata.create_all(bind=engine)

    set_db_connections(mongo_client=connect_to_mongo(),
                       mysql_conn=engine.connect())
    yield
    clear_db_connections()


app = FastAPI(title="Career Guidance API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "https://your-frontend-domain.com",
                   "https://trueyoucareers.com",
                   "https://api.trueyoucareers.com"
                   ],
    # Or, for more permissive (less secure for production if open to public):
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(profile.router, prefix="/profile", tags=["profile"])
app.include_router(guidance.router, prefix="/guidance",
                   tags=["guidance"])
app.include_router(report.router, prefix="/report",
                   tags=["report"])
app.include_router(report_router)
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(question.router, prefix='/questions',tags=['question'])

@app.get("/")
async def root():
    return {"message": "Welcome to the Career Guidance API"}


if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
