import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_best_match(self, query_embedding: List[float], stored_embeddings: List[List[float]]) -> Tuple[float, int]:
        """Find the best matching embedding from a list of stored embeddings"""
        if not stored_embeddings:
            return 0.0, -1
        
        max_similarity = 0.0
        best_match_index = -1
        
        for i, stored_embedding in enumerate(stored_embeddings):
            similarity = self.calculate_similarity(query_embedding, stored_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_index = i
        
        return max_similarity, best_match_index
    
    def match_against_students(self, query_embedding: List[float], students_data: List[dict]) -> Optional[Tuple[dict, float]]:
        """
        Match query embedding against all students and return the best match
        
        Args:
            query_embedding: The face embedding to search for
            students_data: List of student documents with embeddings
            
        Returns:
            Tuple of (student_data, confidence) or None if no match above threshold
        """
        best_student = None
        best_confidence = 0.0
        
        for student in students_data:
            if not student.get('embeddings') or not student.get('is_active', True):
                continue
            
            # Find best match among all embeddings for this student
            max_similarity, _ = self.find_best_match(query_embedding, student['embeddings'])
            
            if max_similarity > best_confidence:
                best_confidence = max_similarity
                best_student = student
        
        # Only return match if confidence is above threshold
        if best_confidence >= self.similarity_threshold:
            return best_student, best_confidence
        
        return None
    
    def set_threshold(self, threshold: float):
        """Update similarity threshold"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"Similarity threshold updated to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def get_threshold(self) -> float:
        """Get current similarity threshold"""
        return self.similarity_threshold
    
    def batch_match(self, query_embedding: List[float], students_data: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Get top K matches for a query embedding
        
        Args:
            query_embedding: The face embedding to search for
            students_data: List of student documents
            top_k: Number of top matches to return
            
        Returns:
            List of (student_data, confidence) tuples sorted by confidence
        """
        matches = []
        
        for student in students_data:
            if not student.get('embeddings') or not student.get('is_active', True):
                continue
            
            max_similarity, _ = self.find_best_match(query_embedding, student['embeddings'])
            
            if max_similarity >= self.similarity_threshold:
                matches.append((student, max_similarity))
        
        # Sort by confidence (descending) and return top K
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

# Global matcher instance
face_matcher = FaceMatcher()
