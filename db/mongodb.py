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
        mongodb.client = AsyncIOMotorClient(MONGODB_URI)
        mongodb.database = mongodb.client[DB_NAME]
        
        # Test the connection
        await mongodb.client.admin.command('ping')
        logging.info("Connected to MongoDB successfully!")
        
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise e

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