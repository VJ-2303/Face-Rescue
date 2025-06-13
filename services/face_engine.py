import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import io
import base64
import logging
from typing import List, Optional, Tuple
import tempfile
import os

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self):
        self.model_name = "VGG-Face"  # Using VGG-Face as it's more stable
        self.detector_backend = "opencv"
        self.confidence_threshold = 0.7
        
    def extract_face_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract face embedding from image bytes
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return None
            
            # Create temporary file for DeepFace
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, img)
                tmp_path = tmp_file.name
            
            try:
                # Extract embedding using DeepFace
                embedding = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True
                )
                
                # DeepFace returns a list of dictionaries, get the first one
                if embedding and len(embedding) > 0:
                    face_embedding = embedding[0]["embedding"]
                    logger.info(f"Successfully extracted embedding of size {len(face_embedding)}")
                    return face_embedding
                else:
                    logger.warning("No face detected in the image")
                    return None
                    
            except Exception as e:
                logger.error(f"DeepFace processing failed: {e}")
                return None
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error in extract_face_embedding: {e}")
            return None
    
    def extract_multiple_embeddings(self, image_files: List[bytes]) -> List[List[float]]:
        """
        Extract embeddings from multiple images
        """
        embeddings = []
        
        for i, image_data in enumerate(image_files):
            logger.info(f"Processing image {i+1}/{len(image_files)}")
            embedding = self.extract_face_embedding(image_data)
            
            if embedding:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to extract embedding from image {i+1}")
        
        logger.info(f"Successfully extracted {len(embeddings)} embeddings from {len(image_files)} images")
        return embeddings
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
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
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_best_match(self, target_embedding: List[float], stored_embeddings: List[List[float]]) -> Tuple[float, int]:
        """
        Find the best matching embedding from stored embeddings
        Returns: (best_similarity, best_index)
        """
        if not stored_embeddings:
            return 0.0, -1
        
        best_similarity = 0.0
        best_index = -1
        
        for i, stored_embedding in enumerate(stored_embeddings):
            similarity = self.calculate_similarity(target_embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        
        return best_similarity, best_index
    
    def is_valid_match(self, similarity: float) -> bool:
        """
        Check if similarity score indicates a valid match
        """
        return similarity >= self.confidence_threshold
    
    def preprocess_image(self, image_data: bytes) -> Optional[bytes]:
        """
        Preprocess image for better face detection
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Resize if image is too large
            height, width = img.shape[:2]
            if width > 1000 or height > 1000:
                scale = min(1000/width, 1000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Enhance image quality
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image_data

# Global instance
face_processor = FaceProcessor()
