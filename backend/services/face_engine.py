import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import List, Optional, Tuple
import logging
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self):
        self.app = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize InsightFace model"""
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            raise e
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Invalid image format: {e}")
    
    def extract_faces(self, image: np.ndarray) -> List[dict]:
        """Extract all faces from an image"""
        try:
            faces = self.app.get(image)
            return faces
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            return []
    
    def get_face_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """Extract face embedding from image"""
        try:
            # Preprocess image
            img = self.preprocess_image(image_data)
            
            # Extract faces
            faces = self.extract_faces(img)
            
            if not faces:
                logger.warning("No faces detected in image")
                return None
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the largest one")
                # Use the face with largest bounding box
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            # Get embedding from the first (largest) face
            embedding = faces[0].embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error getting face embedding: {e}")
            return None
    
    def get_multiple_embeddings(self, image_data_list: List[bytes]) -> List[List[float]]:
        """Extract embeddings from multiple images"""
        embeddings = []
        
        for i, image_data in enumerate(image_data_list):
            embedding = self.get_face_embedding(image_data)
            if embedding:
                embeddings.append(embedding)
                logger.info(f"Successfully extracted embedding from image {i+1}")
            else:
                logger.warning(f"Failed to extract embedding from image {i+1}")
        
        return embeddings
    
    def validate_face_quality(self, image_data: bytes) -> Tuple[bool, str]:
        """Validate if the face in image is of good quality"""
        try:
            img = self.preprocess_image(image_data)
            faces = self.extract_faces(img)
            
            if not faces:
                return False, "No face detected in image"
            
            if len(faces) > 1:
                return False, f"Multiple faces detected ({len(faces)}). Please use image with single face"
            
            face = faces[0]
            
            # Check face size (bounding box area)
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = img.shape[0] * img.shape[1]
            face_ratio = face_area / img_area
            
            if face_ratio < 0.02:  # Face should be at least 2% of image
                return False, "Face is too small in the image"
            
            # Check face detection confidence if available
            if hasattr(face, 'det_score') and face.det_score < 0.5:
                return False, "Face detection confidence is too low"
            
            return True, "Face quality is good"
            
        except Exception as e:
            return False, f"Error validating face quality: {e}"

# Global face engine instance
face_engine = FaceEngine()
