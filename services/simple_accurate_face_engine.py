import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Optional, Tuple, Dict, Any
import tempfile
import os
import time

# Reliable libraries without cmake dependencies
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)

class SimpleAccurateFaceProcessor:
    """
    Simple and Accurate Face Recognition Engine using MediaPipe and OpenCV.
    No cmake or Visual Studio build tools required.
    Focuses on reliability and accuracy over complexity.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.face_size = (160, 160)
        
        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
          # Face Detection (more accurate than OpenCV)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for speed, 1 for better accuracy
            min_detection_confidence=0.3  # Lower threshold for testing
        )
          # Face Mesh for detailed landmarks (468 landmarks)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower threshold for testing
            min_tracking_confidence=0.3
        )
        
        # Fallback OpenCV cascade (backup)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Feature processing
        self.scaler = StandardScaler()
        self.use_pca = True
        self.pca_components = 128  # Reduce dimensionality
        self.pca = None
        
        # Quality assessment parameters
        self.quality_threshold = 0.4
        self.use_quality_filter = True
        
        # Enhanced preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        logger.info("SimpleAccurateFaceProcessor initialized with MediaPipe")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using MediaPipe (primary) and OpenCV (fallback)
        """
        detections = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # Primary: MediaPipe Face Detection
        try:
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                h, w = rgb_image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure valid bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 20 and height > 20:  # Minimum face size
                        detections.append({
                            'bbox': (x, y, width, height),
                            'confidence': detection.score[0],
                            'source': 'mediapipe'
                        })
                        
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}")
        
        # Fallback: OpenCV if no MediaPipe detections
        if not detections:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in faces:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,
                        'source': 'opencv'
                    })
                    
            except Exception as e:
                logger.warning(f"OpenCV detection failed: {e}")
        
        return detections
    
    def extract_face_landmarks(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract detailed face landmarks using MediaPipe Face Mesh
        """
        try:
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB) if len(face_roi.shape) == 3 else face_roi
            if len(rgb_roi.shape) == 2:
                rgb_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_GRAY2RGB)
            
            results = self.face_mesh.process(rgb_roi)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract landmark coordinates as features (468 landmarks * 3 coordinates)
                features = []
                for landmark in landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(features)
                
        except Exception as e:
            logger.debug(f"Landmark extraction failed: {e}")
            
        return None
    
    def assess_face_quality(self, face_image: np.ndarray) -> float:
        """
        Assess face image quality using multiple metrics
        """
        try:
            scores = []
            
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_image
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100, 1.0)
            scores.append(sharpness_score * 0.3)
            
            # 2. Brightness (not too dark or bright)
            brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            scores.append(brightness_score * 0.2)
            
            # 3. Contrast
            contrast = np.std(gray_face)
            contrast_score = min(contrast / 50, 1.0)
            scores.append(contrast_score * 0.2)
            
            # 4. Size adequacy
            min_dim = min(face_image.shape[:2])
            size_score = min(min_dim / 50, 1.0)  # Prefer faces >= 50px
            scores.append(size_score * 0.3)
            
            return sum(scores)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Enhanced face preprocessing for better feature extraction
        """
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            # 1. Histogram equalization for better contrast
            equalized = cv2.equalizeHist(gray_face)
            
            # 2. CLAHE for adaptive contrast enhancement
            clahe_enhanced = self.clahe.apply(equalized)
            
            # 3. Slight Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(clahe_enhanced, (3, 3), 0)
            
            # 4. Resize to standard size
            resized = cv2.resize(blurred, self.face_size, interpolation=cv2.INTER_LANCZOS4)
            
            return resized
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            # Fallback: just resize
            return cv2.resize(face_roi, self.face_size)
    
    def extract_statistical_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from preprocessed face image
        """
        features = []
        
        try:
            # 1. Pixel intensity statistics
            features.extend([
                np.mean(face_image),
                np.std(face_image),
                np.median(face_image),
                np.min(face_image),
                np.max(face_image),
                np.percentile(face_image, 25),
                np.percentile(face_image, 75)
            ])
            
            # 2. Regional statistics (divide face into regions)
            h, w = face_image.shape
            regions = [
                face_image[:h//2, :w//2],      # Top-left
                face_image[:h//2, w//2:],      # Top-right
                face_image[h//2:, :w//2],      # Bottom-left
                face_image[h//2:, w//2:],      # Bottom-right
                face_image[h//4:3*h//4, w//4:3*w//4],  # Center
            ]
            
            for region in regions:
                if region.size > 0:
                    features.extend([
                        np.mean(region),
                        np.std(region)
                    ])
            
            # 3. Gradient features
            grad_x = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
            # 4. Texture features (simplified LBP)
            lbp_features = self.extract_simple_lbp(face_image)
            features.extend(lbp_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Statistical feature extraction failed: {e}")
            return np.zeros(50)  # Return default features
    
    def extract_simple_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> List[float]:
        """
        Extract simplified Local Binary Pattern features
        """
        try:
            # Simple LBP implementation
            rows, cols = image.shape
            lbp = np.zeros_like(image)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    binary_string = ''
                    
                    # Sample points around the center
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = i + radius * np.cos(angle)
                        y = j + radius * np.sin(angle)
                        
                        # Bilinear interpolation
                        x_floor, y_floor = int(x), int(y)
                        if 0 <= x_floor < rows-1 and 0 <= y_floor < cols-1:
                            dx, dy = x - x_floor, y - y_floor
                            pixel_value = (1-dx) * (1-dy) * image[x_floor, y_floor] + \
                                        dx * (1-dy) * image[x_floor+1, y_floor] + \
                                        (1-dx) * dy * image[x_floor, y_floor+1] + \
                                        dx * dy * image[x_floor+1, y_floor+1]
                            
                            binary_string += '1' if pixel_value >= center else '0'
                    
                    if len(binary_string) == n_points:
                        lbp[i, j] = int(binary_string, 2)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=2**n_points, range=(0, 2**n_points))
            
            # Normalize and return
            return (hist / (hist.sum() + 1e-7)).tolist()
            
        except Exception as e:
            logger.warning(f"LBP extraction failed: {e}")
            return [0.0] * (2**n_points)
    
    def extract_face_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract face embedding from image bytes
        """
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return None
            
            # Detect faces
            face_detections = self.detect_faces(image)
            
            if not face_detections:
                logger.warning("No faces detected in image")
                return None
            
            # Select best face (highest confidence or quality)
            best_face = None
            best_score = 0.0
            
            for detection in face_detections:
                x, y, w, h = detection['bbox']
                
                # Extract face region with padding
                padding = int(min(w, h) * 0.1)
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w + padding)
                y_end = min(image.shape[0], y + h + padding)
                
                face_roi = image[y_start:y_end, x_start:x_end]
                
                # Assess quality if enabled
                if self.use_quality_filter:
                    quality_score = self.assess_face_quality(face_roi)
                    combined_score = detection['confidence'] * 0.7 + quality_score * 0.3
                else:
                    combined_score = detection['confidence']
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_face = face_roi
            
            if best_face is None:
                logger.warning("No suitable face found")
                return None
            
            # Check minimum quality
            if self.use_quality_filter and best_score < self.quality_threshold:
                logger.warning(f"Face quality too low: {best_score:.3f} < {self.quality_threshold}")
                return None
            
            # Preprocess the face
            processed_face = self.preprocess_face(best_face)
            
            # Extract features
            features = []
            
            # 1. MediaPipe landmarks (most important)
            landmarks = self.extract_face_landmarks(best_face)
            if landmarks is not None:
                features.extend(landmarks)
            else:
                # Fallback: add zeros for landmark features
                features.extend([0.0] * (468 * 3))  # 468 landmarks * 3 coordinates
            
            # 2. Statistical features
            statistical_features = self.extract_statistical_features(processed_face)
            features.extend(statistical_features)
            
            # 3. Raw pixel features (downsampled)
            pixel_features = cv2.resize(processed_face, (16, 16)).flatten().astype(float) / 255.0
            features.extend(pixel_features)
            
            # Convert to numpy array
            feature_vector = np.array(features)
            
            # Apply PCA if trained
            if self.use_pca and self.pca is not None:
                try:
                    feature_vector = self.pca.transform(feature_vector.reshape(1, -1))[0]
                except:
                    pass  # Use original features if PCA fails
            
            # Normalize features
            feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            
            return feature_vector.tolist()
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compare two face embeddings using multiple similarity metrics
        """
        try:
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            if emb1.shape[1] != emb2.shape[1]:
                logger.warning(f"Embedding dimension mismatch: {emb1.shape[1]} vs {emb2.shape[1]}")
                return 0.0
            
            # Multiple similarity metrics
            similarities = []
            
            # 1. Cosine similarity (primary)
            cosine_sim = cosine_similarity(emb1, emb2)[0, 0]
            similarities.append(cosine_sim * 0.5)
            
            # 2. Euclidean distance similarity
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            max_dist = np.sqrt(emb1.shape[1])
            euclidean_sim = 1.0 - (euclidean_dist / max_dist)
            similarities.append(euclidean_sim * 0.3)
            
            # 3. Pearson correlation
            try:
                correlation = np.corrcoef(emb1.flatten(), emb2.flatten())[0, 1]
                if not np.isnan(correlation):
                    similarities.append(correlation * 0.2)
            except:
                similarities.append(0.0)
            
            # Calculate weighted average
            final_similarity = sum(similarities)
            
            # Apply sigmoid for better separation
            sigmoid_factor = 12
            threshold = 0.5
            enhanced_similarity = 1 / (1 + np.exp(-sigmoid_factor * (final_similarity - threshold)))
            
            return min(max(enhanced_similarity, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return 0.0
    
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
    
    def train_pca(self, embeddings: List[List[float]]):
        """
        Train PCA on a set of embeddings for dimensionality reduction
        """
        try:
            if len(embeddings) < 10:
                logger.warning("Not enough embeddings to train PCA")
                return
            
            X = np.array(embeddings)
            self.pca = PCA(n_components=min(self.pca_components, X.shape[1]))
            self.pca.fit(X)
            
            logger.info(f"PCA trained with {self.pca.n_components_} components, "
                       f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
            
        except Exception as e:
            logger.error(f"PCA training failed: {e}")
    
    def save_model(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'pca': self.pca,
                'scaler': self.scaler,
                'config': {
                    'confidence_threshold': self.confidence_threshold,
                    'quality_threshold': self.quality_threshold,
                    'face_size': self.face_size,
                    'use_pca': self.use_pca,
                    'pca_components': self.pca_components
                }
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained models from disk"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.pca = model_data.get('pca')
                self.scaler = model_data.get('scaler')
                
                # Load config
                config = model_data.get('config', {})
                self.confidence_threshold = config.get('confidence_threshold', 0.6)
                self.quality_threshold = config.get('quality_threshold', 0.4)
                self.face_size = config.get('face_size', (160, 160))
                self.use_pca = config.get('use_pca', True)
                self.pca_components = config.get('pca_components', 128)
                
                logger.info(f"Model loaded from {filepath}")
            else:
                logger.warning(f"Model file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def benchmark(self, test_image_path: str) -> Dict[str, float]:
        """Benchmark processing speed"""
        if not os.path.exists(test_image_path):
            return {'error': 'Test image not found'}
        
        try:
            with open(test_image_path, 'rb') as f:
                image_data = f.read()
            
            # Time face detection
            start_time = time.time()
            embedding = self.extract_face_embedding(image_data)
            total_time = time.time() - start_time
            
            return {
                'total_time': total_time,
                'success': embedding is not None,
                'embedding_size': len(embedding) if embedding else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            'name': 'SimpleAccurateFaceProcessor',
            'version': '1.0',
            'backend': 'MediaPipe + OpenCV',
            'features': [
                'MediaPipe Face Detection',
                'MediaPipe Face Mesh (468 landmarks)',
                'Statistical Features',
                'LBP Texture Features',
                'Quality Assessment',
                'PCA Dimensionality Reduction'
            ],
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'quality_threshold': self.quality_threshold,
                'face_size': self.face_size,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components
            }
        }

# Create global instance
simple_accurate_face_processor = SimpleAccurateFaceProcessor()
