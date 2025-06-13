import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Optional, Tuple
import tempfile
import os
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

logger = logging.getLogger(__name__)

class SimpleFaceProcessor:
    def __init__(self):
        self.confidence_threshold = 0.6  # Increased for better accuracy
        
        # Load multiple face detection cascades for better detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Alternative cascade for profile faces
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        
        # Additional cascade for better detection
        alt_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        self.alt_cascade = cv2.CascadeClassifier(alt_cascade_path)
        
        # Eye cascade for face alignment
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Feature extraction parameters - optimized for accuracy
        self.face_size = (224, 224)  # Larger size for better features
        self.use_histogram_equalization = True
        self.use_gaussian_blur = True
        self.use_clahe = True
        self.use_face_alignment = True
        self.use_quality_filter = True
        self.use_advanced_preprocessing = True
        self.use_ensemble_comparison = True
        
        # Quality thresholds
        self.quality_threshold = 0.3
        
        # Initialize CLAHE with multiple configurations
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.clahe_strong = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
        
        # Gabor filter parameters
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_orientations = [0, 45, 90, 135]
        
        logger.info("Enhanced SimpleFaceProcessor initialized with advanced accuracy features")

    def assess_face_quality(self, face_image):
        """
        Assess face image quality to filter out poor quality faces
        """
        try:
            scores = []
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_image, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100, 1.0)
            scores.append(sharpness_score)
            
            # 2. Brightness (not too dark or bright)
            brightness = np.mean(face_image)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            scores.append(brightness_score)
            
            # 3. Contrast
            contrast = np.std(face_image)
            contrast_score = min(contrast / 50, 1.0)
            scores.append(contrast_score)
            
            # 4. Face symmetry (simple check)
            if face_image.shape[1] > 20:  # Ensure face is wide enough
                left_half = face_image[:, :face_image.shape[1]//2]
                right_half = cv2.flip(face_image[:, face_image.shape[1]//2:], 1)
                
                if left_half.shape == right_half.shape:
                    symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
                    scores.append(max(0.0, symmetry_score))
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error in quality assessment: {str(e)}")
            return 0.5  # Default moderate quality

    def align_face(self, face_image):
        """
        Align face using eye detection for better feature extraction
        """
        try:
            if not self.use_face_alignment:
                return face_image
                
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_image, 1.1, 5, minSize=(10, 10))
            
            if len(eyes) >= 2:
                # Sort eyes by x coordinate (left to right)
                eyes = sorted(eyes, key=lambda x: x[0])
                
                # Get eye centers
                eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
                eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
                
                # Calculate angle between eyes
                dx = eye2_center[0] - eye1_center[0]
                dy = eye2_center[1] - eye1_center[1]
                
                if dx != 0:  # Avoid division by zero
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Only rotate if angle is significant (> 2 degrees)
                    if abs(angle) > 2:
                        center = ((eye1_center[0] + eye2_center[0])//2, (eye1_center[1] + eye2_center[1])//2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        aligned_face = cv2.warpAffine(face_image, rotation_matrix, 
                                                    (face_image.shape[1], face_image.shape[0]),
                                                    flags=cv2.INTER_CUBIC)
                        return aligned_face
            
            return face_image
            
        except Exception as e:
            logger.warning(f"Face alignment failed: {str(e)}")
            return face_image

    def advanced_preprocessing(self, face_roi):
        """
        Advanced preprocessing pipeline for maximum accuracy
        """
        try:
            if not self.use_advanced_preprocessing:
                return self.preprocess_face_image(face_roi)
            
            # 1. Face alignment first
            face_roi = self.align_face(face_roi)
            
            # 2. Noise reduction with bilateral filter
            face_roi = cv2.bilateralFilter(face_roi, 9, 75, 75)
            
            # 3. Multi-scale enhancement using image pyramid
            pyramid = [face_roi]
            temp = face_roi.copy()
            
            # Create pyramid levels
            for i in range(2):
                temp = cv2.pyrDown(temp)
                if temp.shape[0] > 10 and temp.shape[1] > 10:  # Ensure minimum size
                    temp_up = cv2.pyrUp(temp, dstsize=(face_roi.shape[1], face_roi.shape[0]))
                    pyramid.append(temp_up)
            
            # Combine pyramid levels for enhanced detail
            if len(pyramid) > 1:
                enhanced = np.mean(pyramid, axis=0).astype(np.uint8)
            else:
                enhanced = face_roi
            
            # 4. Advanced histogram equalization
            # Apply different CLAHE settings and combine
            enhanced_light = self.clahe.apply(enhanced)
            enhanced_strong = self.clahe_strong.apply(enhanced)
            
            # Weighted combination of different enhancement levels
            final_enhanced = cv2.addWeighted(enhanced_light, 0.7, enhanced_strong, 0.3, 0)
            
            # 5. Gamma correction for better contrast
            gamma = 1.2
            gamma_corrected = np.power(final_enhanced / 255.0, gamma) * 255.0
            final_enhanced = gamma_corrected.astype(np.uint8)
            
            # 6. Additional histogram equalization if needed
            if self.use_histogram_equalization:
                final_enhanced = cv2.equalizeHist(final_enhanced)
              # 7. Final slight blur for noise reduction
            if self.use_gaussian_blur:
                final_enhanced = cv2.GaussianBlur(final_enhanced, (3, 3), 0)
            
            # 8. Resize with high-quality interpolation
            face_resized = cv2.resize(final_enhanced, self.face_size, interpolation=cv2.INTER_LANCZOS4)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {str(e)}")
            # Direct simple preprocessing to avoid infinite recursion
            try:
                # Apply histogram equalization for better contrast
                if self.use_histogram_equalization:
                    face_roi = cv2.equalizeHist(face_roi)
                
                # Apply CLAHE for better local contrast
                if self.use_clahe:
                    face_roi = self.clahe.apply(face_roi)
                
                # Apply slight Gaussian blur to reduce noise
                if self.use_gaussian_blur:
                    face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
                
                # Resize to standard size with better interpolation
                face_resized = cv2.resize(face_roi, self.face_size, interpolation=cv2.INTER_LANCZOS4)
                return face_resized
            except:
                # Ultimate fallback - just resize
                return cv2.resize(face_roi, self.face_size)

    def extract_gabor_features(self, face_image):
        """
        Extract Gabor filter features for enhanced texture analysis
        """
        try:
            features = []
            
            # Convert to float for gabor processing
            face_float = face_image.astype(np.float64) / 255.0
            
            for freq in self.gabor_frequencies:
                for angle_deg in self.gabor_orientations:
                    angle_rad = np.radians(angle_deg)
                    
                    try:
                        # Apply Gabor filter
                        filtered_real, filtered_imag = gabor(face_float, frequency=freq, theta=angle_rad)
                        
                        # Extract magnitude
                        magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
                        
                        # Extract statistical features from magnitude
                        features.extend([
                            np.mean(magnitude),
                            np.std(magnitude),
                            np.var(magnitude),
                            np.median(magnitude),
                            np.percentile(magnitude, 25),
                            np.percentile(magnitude, 75)
                        ])
                        
                    except Exception as e:
                        logger.warning(f"Gabor filter failed for freq={freq}, angle={angle_deg}: {str(e)}")
                        # Add zeros if gabor fails
                        features.extend([0.0] * 6)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Gabor feature extraction failed: {str(e)}")
            # Return zeros if completely failed
            return np.zeros(len(self.gabor_frequencies) * len(self.gabor_orientations) * 6)

    def extract_enhanced_lbp_features(self, face_image):
        """
        Enhanced LBP feature extraction with multiple radii and points
        """
        try:
            features = []
            
            # Multiple LBP configurations for richer features
            lbp_configs = [
                (1, 8),   # radius=1, points=8
                (2, 16),  # radius=2, points=16
                (3, 24),  # radius=3, points=24
            ]
            
            for radius, n_points in lbp_configs:
                try:
                    # Calculate LBP
                    lbp = local_binary_pattern(face_image, n_points, radius, method='uniform')
                    
                    # Calculate histogram
                    n_bins = n_points + 2  # uniform patterns + non-uniform
                    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                    
                    # Normalize histogram
                    hist_norm = hist.astype(float) / (hist.sum() + 1e-7)
                    features.extend(hist_norm)
                    
                except Exception as e:
                    logger.warning(f"LBP failed for radius={radius}, points={n_points}: {str(e)}")
                    # Add zeros if this configuration fails
                    features.extend([0.0] * (n_points + 2))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Enhanced LBP extraction failed: {str(e)}")
            # Fallback to original LBP
            return self.extract_lbp_features(face_image)

    def extract_texture_features(self, face_image):
        """
        Extract additional texture features
        """
        try:
            features = []
            
            # 1. Gradient-based features
            grad_x = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.percentile(gradient_magnitude, 90)
            ])
            
            # 2. Entropy (texture complexity)
            hist, _ = np.histogram(face_image.ravel(), bins=256, range=(0, 256))
            hist_norm = hist / (hist.sum() + 1e-7)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            features.append(entropy)
            
            # 3. Local variance features
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(face_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((face_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            features.extend([
                np.mean(local_variance),
                np.std(local_variance)
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Texture feature extraction failed: {str(e)}")
            return np.zeros(6)  # Return default features

    def detect_face_with_multiple_cascades(self, gray_image):
        """
        Detect faces using multiple cascade classifiers with enhanced parameters
        """
        faces = []
        
        # Apply multiple preprocessing techniques for better detection
        processed_images = []
        
        # 1. Original with CLAHE
        clahe_image = self.clahe.apply(gray_image)
        processed_images.append(clahe_image)
        
        # 2. Histogram equalized version
        hist_eq_image = cv2.equalizeHist(gray_image)
        processed_images.append(hist_eq_image)
        
        # 3. Gamma corrected version
        gamma = 1.5
        gamma_corrected = np.power(gray_image / 255.0, gamma) * 255.0
        processed_images.append(gamma_corrected.astype(np.uint8))
        
        # Try detection on each preprocessed version
        for processed_img in processed_images:
            # Multiple parameter combinations for better detection
            detection_params = [
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
                {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (25, 25)},
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                {'scaleFactor': 1.15, 'minNeighbors': 6, 'minSize': (35, 35)},
                {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (15, 15)},
            ]
            
            # Try frontal face detection with multiple parameters
            for params in detection_params:
                frontal_faces = self.face_cascade.detectMultiScale(
                    processed_img, 
                    scaleFactor=params['scaleFactor'],
                    minNeighbors=params['minNeighbors'],
                    minSize=params['minSize'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(frontal_faces) > 0:
                    faces.extend(frontal_faces)
                    if len(faces) >= 3:  # Stop if we have enough candidates
                        break
            
            # Try alternative cascade if not enough faces found
            if len(faces) < 2:
                for params in detection_params[:3]:  # Use fewer params for alt cascade
                    alt_faces = self.alt_cascade.detectMultiScale(
                        processed_img, 
                        scaleFactor=params['scaleFactor'],
                        minNeighbors=params['minNeighbors'],
                        minSize=params['minSize']
                    )
                    faces.extend(alt_faces)
            
            # Try profile face detection if still not enough faces
            if len(faces) < 2:
                for params in detection_params[:2]:  # Use fewer params for profile
                    profile_faces = self.profile_cascade.detectMultiScale(
                        processed_img, 
                        scaleFactor=params['scaleFactor'],
                        minNeighbors=params['minNeighbors'],
                        minSize=params['minSize']
                    )
                    faces.extend(profile_faces)
            
            # If we found faces, break from preprocessing loop
            if len(faces) > 0:
                break
        
        # Remove duplicate faces that are too close to each other
        if len(faces) > 1:
            faces = self.remove_duplicate_faces(faces)
        
        # If still no faces, try more aggressive detection
        if len(faces) == 0:
            # Very aggressive detection as last resort
            desperate_faces = self.face_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.03, 
                minNeighbors=2,
                minSize=(15, 15),
                maxSize=(300, 300)
            )
            faces.extend(desperate_faces)
        
        return faces
    
    def remove_duplicate_faces(self, faces):
        """
        Remove duplicate faces that are too close to each other
        """
        if len(faces) <= 1:
            return faces
        
        # Convert to list for easier manipulation
        faces_list = list(faces)
        filtered_faces = []
        
        for i, face1 in enumerate(faces_list):
            x1, y1, w1, h1 = face1
            is_duplicate = False
            
            for j, face2 in enumerate(filtered_faces):
                x2, y2, w2, h2 = face2
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                  # If overlap is significant, consider it a duplicate
                if union_area > 0 and overlap_area / union_area > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_faces.append(face1)
        
        return filtered_faces

    def preprocess_face_image(self, face_roi):
        """
        Enhanced preprocessing for better feature extraction - delegates to advanced preprocessing
        """
        if self.use_advanced_preprocessing:
            return self.advanced_preprocessing(face_roi)
        
        # Fallback to simple preprocessing
        # Apply histogram equalization for better contrast
        if self.use_histogram_equalization:
            face_roi = cv2.equalizeHist(face_roi)
        
        # Apply CLAHE for better local contrast
        if self.use_clahe:
            face_roi = self.clahe.apply(face_roi)
        
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
        
        # 1. Enhanced LBP features (multi-scale)
        enhanced_lbp_features = self.extract_enhanced_lbp_features(face_image)
        features.extend(enhanced_lbp_features)
        
        # 2. HOG (Histogram of Oriented Gradients) features
        hog_features = self.extract_hog_features(face_image)
        features.extend(hog_features)
        
        # 3. Gabor filter features
        gabor_features = self.extract_gabor_features(face_image)
        features.extend(gabor_features)
        
        # 4. Enhanced pixel intensity features
        pixel_features = self.extract_pixel_features(face_image)
        features.extend(pixel_features)
        
        # 5. Additional texture features
        texture_features = self.extract_texture_features(face_image)
        features.extend(texture_features)
        
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
            
            # Assess quality of all detected faces and select the best one
            best_face = None
            best_quality = 0.0
            
            for face in faces:
                x, y, w, h = face
                face_roi = gray[y:y+h, x:x+w]
                
                if self.use_quality_filter:
                    quality_score = self.assess_face_quality(face_roi)
                    if quality_score > best_quality and quality_score > self.quality_threshold:
                        best_quality = quality_score
                        best_face = face
                else:
                    # If not using quality filter, just use the largest face
                    if best_face is None or (w * h) > (best_face[2] * best_face[3]):
                        best_face = face
            
            # If no high-quality face found, use the largest one
            if best_face is None:
                best_face = max(faces, key=lambda face: face[2] * face[3])
                logger.warning(f"No high-quality face found (quality threshold: {self.quality_threshold}), using largest face")
            
            x, y, w, h = best_face
            
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
    
    def ensemble_face_comparison(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Use ensemble of different comparison methods for maximum accuracy
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            if len(emb1) != len(emb2):
                logger.warning(f"Embedding length mismatch: {len(emb1)} vs {len(emb2)}")
                return 0.0
            
            scores = []
            
            # Method 1: Direct cosine similarity (avoid recursion)
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            cosine_score = np.dot(emb1_norm, emb2_norm)
            scores.append(cosine_score)
            
            # Method 2: Feature-specific comparison
            # Assume features are organized as: LBP, HOG, Gabor, Pixel, Texture
            feature_sections = self.split_feature_vector(emb1)
            feature_sections2 = self.split_feature_vector(emb2)
            
            for i, (feat1, feat2) in enumerate(zip(feature_sections, feature_sections2)):
                if len(feat1) > 0 and len(feat2) > 0:
                    feat_score = self.compare_feature_subset(feat1, feat2)
                    scores.append(feat_score)
            
            # Method 3: Structural similarity
            structural_score = self.structural_similarity(emb1, emb2)
            scores.append(structural_score)
            
            # Method 4: Distribution-based comparison
            dist_score = self.distribution_similarity(emb1, emb2)
            scores.append(dist_score)
            
            # Weighted ensemble - give more weight to methods that perform better
            weights = [0.4, 0.15, 0.15, 0.15, 0.1, 0.05]  # Adjust based on number of scores
            weights = weights[:len(scores)]  # Trim to actual number of scores
            weights = np.array(weights) / np.sum(weights)  # Normalize
            
            final_score = np.average(scores, weights=weights)
            
            # Apply adaptive threshold based on score variance
            score_variance = np.var(scores)
            adaptive_factor = 1.0 + 0.5 * score_variance  # Higher variance = more conservative
            
            # Enhanced sigmoid transformation
            sigmoid_factor = 20 * adaptive_factor
            threshold = 0.5 + 0.1 * score_variance
            enhanced_similarity = 1 / (1 + np.exp(-sigmoid_factor * (final_score - threshold)))
            
            return min(max(enhanced_similarity, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in ensemble face comparison: {str(e)}")
            # Direct fallback to avoid infinite recursion
            try:
                emb1 = np.array(embedding1)
                emb2 = np.array(embedding2)
                
                if len(emb1) != len(emb2):
                    return 0.0
                
                # Simple cosine similarity fallback
                emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                cosine_similarity = np.dot(emb1_norm, emb2_norm)
                return max(0.0, min(1.0, cosine_similarity))
            except:
                return 0.0

    def split_feature_vector(self, features):
        """
        Split the feature vector into different feature type sections
        """
        try:
            sections = []
            start_idx = 0
            
            # Approximate feature sizes (adjust based on your actual implementation)
            feature_sizes = {
                'lbp': 50,      # Enhanced LBP features
                'hog': 100,     # HOG features  
                'gabor': 72,    # Gabor features (3 freq * 4 orient * 6 stats)
                'texture': 6,   # Texture features
                'pixel': -1     # Remaining features are pixel features
            }
            
            for feature_type, size in feature_sizes.items():
                if feature_type == 'pixel':
                    # Take remaining features
                    sections.append(features[start_idx:])
                else:
                    end_idx = start_idx + size
                    if end_idx <= len(features):
                        sections.append(features[start_idx:end_idx])
                        start_idx = end_idx
                    else:
                        sections.append(features[start_idx:])
                        break
            
            return sections
            
        except Exception as e:
            logger.warning(f"Error splitting feature vector: {str(e)}")
            return [features]  # Return whole vector if splitting fails

    def compare_feature_subset(self, feat1, feat2):
        """
        Compare a subset of features using multiple metrics
        """
        try:
            # Normalize features
            feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
            feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
            
            # Multiple similarity metrics
            cosine_sim = np.dot(feat1_norm, feat2_norm)
            
            # Histogram intersection (for histogram-based features like LBP)
            hist_intersect = np.sum(np.minimum(feat1, feat2)) / (np.sum(feat1) + 1e-8)
            
            # Combined score
            combined = 0.7 * cosine_sim + 0.3 * hist_intersect
            
            return max(0.0, min(1.0, combined))
            
        except Exception as e:
            logger.warning(f"Error in feature subset comparison: {str(e)}")
            return 0.0

    def structural_similarity(self, emb1, emb2):
        """
        Compare structural patterns in embeddings
        """
        try:
            # Compute local patterns using sliding window
            window_size = min(10, len(emb1) // 10)
            if window_size < 3:
                return 0.5
            
            similarities = []
            
            for i in range(0, len(emb1) - window_size + 1, window_size):
                window1 = emb1[i:i+window_size]
                window2 = emb2[i:i+window_size]
                
                # Local correlation
                if np.std(window1) > 1e-6 and np.std(window2) > 1e-6:
                    corr = np.corrcoef(window1, window2)[0, 1]
                    if not np.isnan(corr):
                        similarities.append(abs(corr))
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Error in structural similarity: {str(e)}")
            return 0.5

    def distribution_similarity(self, emb1, emb2):
        """
        Compare statistical distributions of embeddings
        """
        try:
            # Statistical moments comparison
            moments1 = [np.mean(emb1), np.std(emb1), np.mean(emb1**3), np.mean(emb1**4)]
            moments2 = [np.mean(emb2), np.std(emb2), np.mean(emb2**3), np.mean(emb2**4)]
              # Normalized difference
            moment_diffs = []
            for m1, m2 in zip(moments1, moments2):
                if abs(m1) + abs(m2) > 1e-8:
                    diff = 1.0 - abs(m1 - m2) / (abs(m1) + abs(m2))
                    moment_diffs.append(max(0.0, diff))
            
            return np.mean(moment_diffs) if moment_diffs else 0.5
            
        except Exception as e:
            logger.warning(f"Error in distribution similarity: {str(e)}")
            return 0.5
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compare two face embeddings using enhanced similarity metrics with ensemble approach
        """
        try:
            # Use ensemble comparison if enabled, otherwise fall back to standard comparison
            if self.use_ensemble_comparison:
                return self.ensemble_face_comparison(embedding1, embedding2)
            
            # Standard comparison (fallback)
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Ensure embeddings have the same length
            if len(emb1) != len(emb2):
                logger.warning(f"Embedding length mismatch: {len(emb1)} vs {len(emb2)}")
                return 0.0
            
            # Normalize embeddings for better comparison
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # 1. Cosine similarity (most important for face recognition)
            cosine_similarity = np.dot(emb1_norm, emb2_norm)
            
            # 2. Pearson correlation coefficient
            correlation = np.corrcoef(emb1, emb2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 3. Chi-square distance (good for histogram features)
            chi_square_sim = 0.0
            for i in range(len(emb1)):
                if emb1[i] + emb2[i] > 0:
                    chi_square_sim += ((emb1[i] - emb2[i]) ** 2) / (emb1[i] + emb2[i])
            chi_square_sim = 1.0 / (1.0 + chi_square_sim / len(emb1))
            
            # 4. Manhattan distance similarity
            manhattan_dist = np.sum(np.abs(emb1_norm - emb2_norm))
            manhattan_sim = 1.0 / (1.0 + manhattan_dist)
            
            # 5. Weighted combination of different similarity measures
            # Cosine similarity is most important for face recognition
            combined_similarity = (
                0.6 * cosine_similarity +      # Primary metric
                0.2 * correlation +            # Secondary metric
                0.1 * chi_square_sim +         # Good for LBP features
                0.1 * manhattan_sim            # Additional metric
            )
            
            # Apply sigmoid function to enhance separation between matches and non-matches
            # This helps create clearer decision boundaries
            sigmoid_factor = 15  # Controls the steepness of the sigmoid
            enhanced_similarity = 1 / (1 + np.exp(-sigmoid_factor * (combined_similarity - 0.5)))
            
            # Convert to confidence (0-1 range) with better scaling
            confidence = min(max(enhanced_similarity, 0.0), 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return 0.0

# Create global instance
simple_face_processor = SimpleFaceProcessor()