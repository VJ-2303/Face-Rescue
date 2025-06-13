import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Optional, Tuple
import tempfile
import os

logger = logging.getLogger(__name__)

class EnhancedFaceProcessor:
    def __init__(self):
        self.confidence_threshold = 0.65  # Lowered slightly for better matches
        
        # Load multiple face detection cascades for better detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Alternative cascade for profile faces
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        
        # Feature extraction parameters
        self.face_size = (128, 128)  # Increased from 64x64 for better features
        self.use_histogram_equalization = True
        self.use_gaussian_blur = True
        
    def detect_face_with_multiple_cascades(self, gray_image):
        """
        Detect faces using multiple cascade classifiers for better accuracy
        """
        faces = []
        
        # Try frontal face detection with multiple parameters
        frontal_faces = self.face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(frontal_faces) > 0:
            faces.extend(frontal_faces)
        
        # Try with different parameters if no faces found
        if len(faces) == 0:
            frontal_faces = self.face_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.05, 
                minNeighbors=3,
                minSize=(20, 20)
            )
            faces.extend(frontal_faces)
        
        # Try profile face detection if still no faces
        if len(faces) == 0:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.1, 
                minNeighbors=4,
                minSize=(30, 30)
            )
            faces.extend(profile_faces)
        
        return faces
    
    def preprocess_face_image(self, face_roi):
        """
        Enhanced preprocessing for better feature extraction
        """
        # Apply histogram equalization for better contrast
        if self.use_histogram_equalization:
            face_roi = cv2.equalizeHist(face_roi)
        
        # Apply slight Gaussian blur to reduce noise
        if self.use_gaussian_blur:
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # Resize to standard size with better interpolation
        face_resized = cv2.resize(face_roi, self.face_size, interpolation=cv2.INTER_LANCZOS4)
        
        return face_resized
    
    def extract_advanced_features(self, face_image):
        """
        Extract multiple types of features for better representation
        """
        features = []
        
        # 1. LBP (Local Binary Pattern) features
        lbp_features = self.extract_lbp_features(face_image)
        features.extend(lbp_features)
        
        # 2. HOG (Histogram of Oriented Gradients) features
        hog_features = self.extract_hog_features(face_image)
        features.extend(hog_features)
        
        # 3. Enhanced pixel intensity features
        pixel_features = self.extract_pixel_features(face_image)
        features.extend(pixel_features)
        
        return np.array(features)
    
    def extract_lbp_features(self, face_image):
        """
        Extract Local Binary Pattern features
        """
        # Calculate LBP using a simplified approach
        radius = 2
        n_points = 8
        
        lbp = np.zeros_like(face_image)
        for i in range(radius, face_image.shape[0] - radius):
            for j in range(radius, face_image.shape[1] - radius):
                center = face_image[i, j]
                binary_val = 0
                
                # Sample 8 points around the center
                neighbors = [
                    face_image[i-radius, j-radius], face_image[i-radius, j], face_image[i-radius, j+radius],
                    face_image[i, j+radius], face_image[i+radius, j+radius], face_image[i+radius, j],
                    face_image[i+radius, j-radius], face_image[i, j-radius]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary_val += 2**k
                
                lbp[i, j] = binary_val
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
    
    def extract_hog_features(self, face_image):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        """
        # Calculate gradients
        grad_x = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Divide image into cells and calculate histogram for each cell
        cell_size = 16
        n_bins = 9
        hog_features = []
        
        for i in range(0, face_image.shape[0] - cell_size + 1, cell_size):
            for j in range(0, face_image.shape[1] - cell_size + 1, cell_size):
                cell_magnitude = magnitude[i:i+cell_size, j:j+cell_size]
                cell_direction = direction[i:i+cell_size, j:j+cell_size]
                
                # Calculate histogram
                hist, _ = np.histogram(cell_direction.ravel(), bins=n_bins, 
                                    range=(-np.pi, np.pi), weights=cell_magnitude.ravel())
                hog_features.extend(hist)
        
        return np.array(hog_features)
    
    def extract_pixel_features(self, face_image):
        """
        Extract enhanced pixel-based features
        """
        # Divide face into regions and extract statistical features
        h, w = face_image.shape
        regions = [
            face_image[:h//2, :w//2],  # Top-left
            face_image[:h//2, w//2:],  # Top-right
            face_image[h//2:, :w//2],  # Bottom-left
            face_image[h//2:, w//2:],  # Bottom-right
            face_image[h//4:3*h//4, w//4:3*w//4],  # Center region
        ]
        
        features = []
        for region in regions:
            # Statistical features for each region
            features.extend([
                np.mean(region),
                np.std(region),
                np.median(region),
                np.min(region),
                np.max(region)
            ])
        
        # Add normalized pixel intensities (downsampled)
        face_small = cv2.resize(face_image, (32, 32))
        pixel_features = face_small.flatten().astype(float) / 255.0
        features.extend(pixel_features)
        
        return np.array(features)
        
    def extract_face_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract enhanced face features from image bytes
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
            
            # Detect faces using multiple methods
            faces = self.detect_face_with_multiple_cascades(gray)
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return None
            
            # Get the largest face (most prominent)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Add padding around face for better feature extraction
            padding = int(min(w, h) * 0.2)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            face_roi = gray[y_start:y_end, x_start:x_end]
            
            # Preprocess the face image
            face_processed = self.preprocess_face_image(face_roi)
            
            # Extract advanced features
            features = self.extract_advanced_features(face_processed)
            
            return features.tolist()
            
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
        Compare two face embeddings using multiple similarity metrics
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Ensure embeddings have the same length
            if len(emb1) != len(emb2):
                logger.warning(f"Embedding length mismatch: {len(emb1)} vs {len(emb2)}")
                return 0.0
            
            # 1. Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                cosine_similarity = 0.0
            else:
                cosine_similarity = dot_product / (norm1 * norm2)
            
            # 2. Euclidean distance similarity
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            max_possible_dist = np.sqrt(len(emb1))  # Approximate max distance
            euclidean_similarity = 1.0 - (euclidean_dist / max_possible_dist)
            
            # 3. Correlation coefficient
            correlation = np.corrcoef(emb1, emb2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Combine similarities with weights
            combined_similarity = (
                0.5 * cosine_similarity +
                0.3 * euclidean_similarity +
                0.2 * correlation
            )
            
            # Convert to confidence (0-1 range)
            confidence = (combined_similarity + 1) / 2
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return 0.0

# Create global instance
enhanced_face_processor = EnhancedFaceProcessor()
