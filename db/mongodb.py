from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# MongoDB connection settings
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "face_recognition")

# Async MongoDB client for FastAPI
class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    try:
        # Use local MongoDB if no URI is provided
        if not MONGODB_URI:
            mongodb_uri = "mongodb://localhost:27017"
            logging.info("No MONGODB_URI found, using local MongoDB")
        else:
            mongodb_uri = MONGODB_URI
            
        mongodb.client = AsyncIOMotorClient(mongodb_uri)
        mongodb.database = mongodb.client[DB_NAME]
        
        # Test the connection
        await mongodb.client.admin.command('ping')
        logging.info("Connected to MongoDB successfully!")
        
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        # For development, continue without MongoDB
        logging.warning("Continuing without MongoDB connection for development")
        mongodb.client = None
        mongodb.database = None

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        logging.info("Disconnected from MongoDB")

def get_database():
    """Get database instance"""
    return mongodb.database

# Synchronous client for one-time operations
def get_sync_database():
    """Get synchronous database connection"""
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME]