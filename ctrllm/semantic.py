"""
Semantic Analysis Module
Contains metrics related to semantic and meaning-based analysis
"""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class SemanticAnalyzer:
    """
    Analyzes semantic properties of text including:
    - Embedding variance (semantic diversity)
    - Sentence similarity
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic analyzer.
        
        Args:
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.sentences = []
        self.embeddings = None
        
    def set_sentences(self, sentences: List[str]):
        """
        Set sentences for analysis.
        
        Args:
            sentences: List of sentence strings
        """
        self.sentences = sentences
        self.embeddings = None  # Reset embeddings
        
    def compute_embeddings(self) -> np.ndarray:
        """
        Compute sentence embeddings.
        
        Returns:
            Array of sentence embeddings with shape (N, embedding_dim)
        """
        if not self.sentences:
            raise ValueError("No sentences set. Call set_sentences() first.")
            
        self.embeddings = self.embedding_model.encode(self.sentences)
        return self.embeddings
    
    def embedding_variance(self) -> float:
        """
        Calculate variance of sentence embeddings in semantic space.
        Formula: σ²_emb = (1/N) Σ ||e_i - e_mean||²
        
        Returns:
            Variance of embeddings
        """
        if self.embeddings is None:
            self.compute_embeddings()
            
        mean_embedding = np.mean(self.embeddings, axis=0)
        variance = np.mean([np.linalg.norm(emb - mean_embedding)**2 
                           for emb in self.embeddings])
        return float(variance)
    
    def pairwise_similarity(self) -> Dict[str, float]:
        """
        Calculate pairwise cosine similarity between sentences.
        
        Returns:
            Dictionary with mean, min, max similarity scores
        """
        if self.embeddings is None:
            self.compute_embeddings()
        
        if len(self.embeddings) < 2:
            return {
                'mean_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        similarities = []
        n = len(self.embeddings)
        
        for i in range(n):
            for j in range(i+1, n):
                # Cosine similarity
                sim = np.dot(self.embeddings[i], self.embeddings[j]) / (
                    np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j])
                )
                similarities.append(sim)
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'num_pairs': len(similarities)
        }
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Compute all semantic metrics.
        
        Returns:
            Dictionary with all semantic metrics
        """
        return {
            'embedding_variance': self.embedding_variance(),
            'pairwise_similarity': self.pairwise_similarity()
        }
