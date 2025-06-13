from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os
from contextlib import asynccontextmanager

from db.mongodb import connect_to_mongo, close_mongo_connection
from routers import register, search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Face Recognition API...")
    await connect_to_mongo()
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Recognition API...")
    await close_mongo_connection()

# Create FastAPI app
app = FastAPI(
    title="Missing Children Face Recognition API",
    description="AI-powered face recognition system for identifying missing/special children",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(register.router)
app.include_router(search.router)

# Serve static files (frontend)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """
    Root endpoint - serve frontend
    """
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {
        "message": "Face Recognition API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "register_student": "/api/students/register",
            "search_face": "/api/search/face",
            "list_students": "/api/students/list",
            "search_stats": "/api/search/stats"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Face Recognition API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8081))
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )