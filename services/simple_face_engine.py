import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Optional, Tuple
import tempfile
import os

logger = logging.getLogger(__name__)

class SimpleFaceProcessor:
    def __init__(self):
        self.confidence_threshold = 0.7
        # Load OpenCV's face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def extract_face_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract simple face features from image bytes (simplified version)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return None
            
            # Take the first (largest) face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Create simple feature vector (flattened pixel values normalized)
            embedding = face_resized.flatten().astype(float) / 255.0
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None
    
    def extract_multiple_embeddings(self, image_data_list: List[bytes]) -> List[List[float]]:
        """
        Extract embeddings from multiple images
        """
        embeddings = []
        for image_data in image_data_list:
            embedding = self.extract_face_embedding(image_data)
            if embedding:
                embeddings.append(embedding)
        return embeddings
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compare two face embeddings using cosine similarity
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert similarity to confidence (0-1 range)
            confidence = (similarity + 1) / 2
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return 0.0

# Create global instance
simple_face_processor = SimpleFaceProcessor()
