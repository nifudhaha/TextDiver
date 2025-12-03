"""
Argument Analysis Module
Contains metrics related to argument detection, clustering, and diversity

Uses LLM-based argument detection for high accuracy.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
import stanza

# OpenAI for LLM-based detection
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# LLM Prompt for argument detection
ARGUMENT_DETECTION_PROMPT = """You are an annotator for argument detection.

Task:
Definition (must satisfy BOTH):
  1) At least one SUBJECTIVE claim/conclusion (opinion, evaluation, stance), AND
  2) At least one supporting sentence giving REASONING, PREMISE, or EVIDENCE
     (objective info, reliable facts, or commonsense knowledge).

Notes:
- Pure descriptions of facts without a stance are NOT arguments.
- Pure rhetorical questions without support are NOT arguments.
- A single subjective claim WITHOUT explicit support is NOT an argument.
- Output STRICT JSON with fields:
  - is_argument (boolean)
  - explanation (string)  # briefly justify your decision
"""


class ArgumentAnalyzer:
    """
    Analyzes argument-related properties of text including:
    - Argument detection (LLM-based)
    - Argument clustering
    - Argument diversity
    - Main vs. Fringe perspective
    - Argument distinctness
    """
    
    def __init__(self, 
                 lang: str = "en",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 min_cluster_size: int = 2,
                 llm_model: str = "gpt-4o-mini",
                 llm_temperature: float = 1.0):
        """
        Initialize argument analyzer.
        
        Args:
            lang: Language code for Stanza (used for sentence tokenization)
            embedding_model: SentenceTransformer model name
            min_cluster_size: Minimum size for a valid cluster
            llm_model: OpenAI model name for argument detection
            llm_temperature: Temperature for LLM (0.0-2.0)
        """
        # Check LLM availability
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        # NLP processing (for sentence tokenization)
        try:
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize',
                download_method=None
            )
        except Exception:
            print(f"Downloading Stanza {lang} models...")
            stanza.download(lang)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize'
            )
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.min_cluster_size = min_cluster_size
        
        # LLM setup
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_client = OpenAI()
        self.llm_cache: Dict[str, Dict] = {}
        
        print(f"ArgumentAnalyzer initialized with LLM model: {llm_model}")
        
        self.doc = None
        self.sentences = []
        self.arguments = []
        self.argument_embeddings = None
        self.clusters = None
        self.cluster_labels = None
        
    def process_text(self, text: str):
        """
        Process text with Stanza (sentence tokenization).
        
        Args:
            text: Input text to analyze
        """
        self.doc = self.nlp(text)
        self.sentences = [sent.text.strip() for sent in self.doc.sentences 
                         if sent.text.strip()]
        
    def detect_arguments(self, batch_delay: float = 0.05, max_retries: int = 3) -> List[str]:
        """
        Detect arguments using LLM (OpenAI API).
        
        Args:
            batch_delay: Delay between API calls (seconds)
            max_retries: Maximum retries for failed API calls
            
        Returns:
            List of argument sentences
        """
        self.arguments = []
        
        for i, sent in enumerate(self.sentences):
            # Check cache first
            cache_key = sent.strip()
            if cache_key in self.llm_cache:
                if self.llm_cache[cache_key]['is_argument']:
                    self.arguments.append(sent)
                continue
            
            # Call LLM
            for attempt in range(max_retries):
                try:
                    messages = [
                        {"role": "system", "content": ARGUMENT_DETECTION_PROMPT},
                        {"role": "user", "content": f"UNIT: {sent}"}
                    ]
                    
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=self.llm_temperature,
                        response_format={"type": "json_object"}
                    )
                    
                    raw = response.choices[0].message.content.strip()
                    result = json.loads(raw)
                    
                    is_arg = bool(result.get('is_argument', False))
                    explanation = str(result.get('explanation', ''))
                    
                    # Cache result
                    self.llm_cache[cache_key] = {
                        'is_argument': is_arg,
                        'explanation': explanation
                    }
                    
                    if is_arg:
                        self.arguments.append(sent)
                    
                    # Rate limiting
                    time.sleep(batch_delay)
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to classify sentence {i+1}/{len(self.sentences)}: {e}")
                        # Default to False if all retries fail
                        self.llm_cache[cache_key] = {
                            'is_argument': False,
                            'explanation': f'Error: {str(e)}'
                        }
                    else:
                        time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return self.arguments
    def compute_argument_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for detected arguments.
        
        Returns:
            Array of argument embeddings with shape (N_args, embedding_dim)
        """
        if not self.arguments:
            raise ValueError("No arguments detected. Call detect_arguments_heuristic() first.")
        
        self.argument_embeddings = self.embedding_model.encode(self.arguments)
        return self.argument_embeddings
    
    def cluster_arguments(self, method: str = 'dbscan', n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Cluster arguments based on their embeddings.
        
        Args:
            method: Clustering method ('dbscan' or 'kmeans')
            n_clusters: Number of clusters for kmeans (auto-detected for dbscan)
            
        Returns:
            Array of cluster labels
        """
        if self.argument_embeddings is None:
            self.compute_argument_embeddings()
        
        if len(self.arguments) < 2:
            # Not enough arguments to cluster
            self.cluster_labels = np.array([0] * len(self.arguments))
            return self.cluster_labels
        
        if method == 'dbscan':
            # DBSCAN - automatically determines number of clusters
            # eps is tuned for normalized embeddings
            clustering = DBSCAN(eps=0.5, min_samples=self.min_cluster_size, metric='cosine')
            self.cluster_labels = clustering.fit_predict(self.argument_embeddings)
            
            # Remove noise points (label = -1) by assigning them to nearest cluster
            if -1 in self.cluster_labels:
                noise_indices = np.where(self.cluster_labels == -1)[0]
                valid_indices = np.where(self.cluster_labels != -1)[0]
                
                if len(valid_indices) > 0:
                    for idx in noise_indices:
                        # Find nearest valid cluster
                        distances = np.linalg.norm(
                            self.argument_embeddings[valid_indices] - self.argument_embeddings[idx],
                            axis=1
                        )
                        nearest_valid_idx = valid_indices[np.argmin(distances)]
                        self.cluster_labels[idx] = self.cluster_labels[nearest_valid_idx]
                else:
                    # All are noise - assign all to cluster 0
                    self.cluster_labels = np.zeros(len(self.cluster_labels), dtype=int)
        
        elif method == 'kmeans':
            if n_clusters is None:
                # Auto-determine number of clusters (rule of thumb: sqrt(n/2))
                n_clusters = max(2, int(np.sqrt(len(self.arguments) / 2)))
            
            n_clusters = min(n_clusters, len(self.arguments))
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = clustering.fit_predict(self.argument_embeddings)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Store cluster information
        self.clusters = {}
        for label in set(self.cluster_labels):
            indices = np.where(self.cluster_labels == label)[0]
            self.clusters[label] = {
                'arguments': [self.arguments[i] for i in indices],
                'embeddings': self.argument_embeddings[indices],
                'size': len(indices)
            }
        
        return self.cluster_labels
    
    def main_vs_fringe_perspective(self) -> Dict[str, float]:
        """
        Calculate main vs. fringe perspective ratio.
        Formula: Main_ratio = max_c nc / Σ nc
        
        Returns:
            Dictionary with main perspective metrics
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments() first.")
        
        if len(self.clusters) == 0:
            return {
                'main_ratio': 0.0,
                'main_cluster_size': 0,
                'total_arguments': 0,
                'num_clusters': 0,
                'fringe_clusters': 0
            }
        
        # Get cluster sizes
        cluster_sizes = [cluster['size'] for cluster in self.clusters.values()]
        total_args = sum(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        
        # Main ratio
        main_ratio = max_cluster_size / total_args if total_args > 0 else 0.0
        
        # Count fringe clusters (smaller than main)
        fringe_clusters = sum(1 for size in cluster_sizes if size < max_cluster_size)
        
        return {
            'main_ratio': float(main_ratio),
            'main_cluster_size': int(max_cluster_size),
            'total_arguments': int(total_args),
            'num_clusters': len(self.clusters),
            'fringe_clusters': int(fringe_clusters),
            'fringe_ratio': float(1 - main_ratio)
        }
    
    def argument_diversity(self) -> Dict[str, float]:
        """
        Calculate argument diversity.
        Formula: Arg_diversity = K / log(1 + Narg)
        
        Returns:
            Dictionary with diversity metrics
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments() first.")
        
        n_args = len(self.arguments)
        k_clusters = len(self.clusters)
        
        if n_args == 0:
            return {
                'arg_diversity': 0.0,
                'num_arguments': 0,
                'num_clusters': 0,
                'cluster_entropy': 0.0
            }
        
        # Basic diversity metric
        diversity = k_clusters / np.log(1 + n_args)
        
        # Additional: cluster size distribution entropy
        cluster_sizes = [cluster['size'] for cluster in self.clusters.values()]
        cluster_probs = np.array(cluster_sizes) / sum(cluster_sizes)
        cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10))
        
        return {
            'arg_diversity': float(diversity),
            'num_arguments': int(n_args),
            'num_clusters': int(k_clusters),
            'cluster_entropy': float(cluster_entropy),
            'normalized_diversity': float(diversity / np.log2(n_args + 1)) if n_args > 0 else 0.0
        }
    
    def argument_distinctness(self) -> Dict[str, float]:
        """
        Calculate argument distinctness (Narrative Distinctness).
        Formula: ND = √(d̄ · dmin)
        
        where:
        - d̄ is the mean pairwise cosine distance between cluster centroids
        - dmin is the minimum pairwise distance
        
        Returns:
            Dictionary with distinctness metrics
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments() first.")
        
        if len(self.clusters) < 2:
            return {
                'narrative_distinctness': 0.0,
                'mean_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'num_clusters': len(self.clusters)
            }
        
        # Compute cluster centroids
        centroids = []
        for cluster in self.clusters.values():
            centroid = np.mean(cluster['embeddings'], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute pairwise cosine distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                # Cosine distance = 1 - cosine similarity
                cos_sim = np.dot(centroids[i], centroids[j]) / (
                    np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                )
                distance = 1 - cos_sim
                distances.append(distance)
        
        if not distances:
            return {
                'narrative_distinctness': 0.0,
                'mean_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'num_clusters': len(self.clusters)
            }
        
        mean_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Narrative Distinctness (ND)
        nd = np.sqrt(mean_dist * min_dist)
        
        return {
            'narrative_distinctness': float(nd),
            'mean_distance': float(mean_dist),
            'min_distance': float(min_dist),
            'max_distance': float(max_dist),
            'num_clusters': len(self.clusters),
            'std_distance': float(np.std(distances))
        }
    
    def deliberation_intensity(self) -> Dict[str, float]:
        """
        Calculate deliberation intensity.
        
        Formula: Delib_intensity = (Ddiv + Dnd) / 2
        
        where:
        - Ddiv is normalized argument diversity [0, 1]
        - Dnd is normalized narrative distinctness [0, 1]
        
        This metric combines argument diversity and distinctness to measure
        the overall quality of deliberation in the text.
        
        Returns:
            Dictionary with deliberation intensity metrics
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments() first.")
        
        # Get diversity and distinctness
        diversity = self.argument_diversity()
        distinctness = self.argument_distinctness()
        
        # Normalize diversity (already has normalized_diversity)
        ddiv = diversity.get('normalized_diversity', 0.0)
        
        # Normalize distinctness (ND is already in [0, 1] range approximately)
        # ND = sqrt(mean * min) where mean, min ∈ [0, 1] for cosine distance
        dnd = distinctness.get('narrative_distinctness', 0.0)
        
        # Deliberation intensity
        delib_intensity = (ddiv + dnd) / 2.0
        
        return {
            'deliberation_intensity': float(delib_intensity),
            'diversity_component': float(ddiv),
            'distinctness_component': float(dnd),
            'interpretation': self._interpret_deliberation(delib_intensity)
        }
    
    def _interpret_deliberation(self, intensity: float) -> str:
        """
        Provide interpretation of deliberation intensity score.
        
        Args:
            intensity: Deliberation intensity value [0, 1]
            
        Returns:
            Interpretation string
        """
        if intensity >= 0.7:
            return "High deliberation quality - diverse and distinct arguments"
        elif intensity >= 0.5:
            return "Moderate deliberation quality - some diversity and distinction"
        elif intensity >= 0.3:
            return "Low deliberation quality - limited diversity or distinction"
        else:
            return "Very low deliberation quality - minimal argumentative structure"
    
    def get_all_metrics(self, clustering_method: str = 'dbscan', **detection_kwargs) -> Dict:
        """
        Compute all argument metrics.
        
        Args:
            clustering_method: 'dbscan' or 'kmeans'
            **detection_kwargs: Additional arguments for argument detection
                batch_delay: Delay between API calls (default: 0.05)
                max_retries: Maximum retries (default: 3)
            
        Returns:
            Dictionary with all argument metrics
        """
        # Step 1: Detect arguments using LLM
        self.detect_arguments(**detection_kwargs)
        
        if len(self.arguments) == 0:
            return {
                'num_arguments': 0,
                'num_sentences': len(self.sentences),
                'argumentativeness': 0.0,
                'llm_model': self.llm_model,
                'main_vs_fringe': {
                    'main_ratio': 0.0,
                    'num_clusters': 0
                },
                'argument_diversity': {
                    'arg_diversity': 0.0,
                    'num_clusters': 0
                },
                'argument_distinctness': {
                    'narrative_distinctness': 0.0,
                    'num_clusters': 0
                },
                'deliberation_intensity': {
                    'deliberation_intensity': 0.0,
                    'interpretation': 'No arguments detected'
                }
            }
        
        # Step 2: Embed arguments
        self.compute_argument_embeddings()
        
        # Step 3: Cluster arguments
        self.cluster_arguments(method=clustering_method)
        
        # Step 4: Compute metrics
        main_fringe = self.main_vs_fringe_perspective()
        diversity = self.argument_diversity()
        distinctness = self.argument_distinctness()
        delib_intensity = self.deliberation_intensity()
        
        # Argumentativeness
        argumentativeness = len(self.arguments) / len(self.sentences) if len(self.sentences) > 0 else 0.0
        
        return {
            'num_arguments': len(self.arguments),
            'num_sentences': len(self.sentences),
            'argumentativeness': float(argumentativeness),
            'llm_model': self.llm_model,
            'arguments': self.arguments,  # Include actual arguments for inspection
            'main_vs_fringe': main_fringe,
            'argument_diversity': diversity,
            'argument_distinctness': distinctness,
            'deliberation_intensity': delib_intensity
        }