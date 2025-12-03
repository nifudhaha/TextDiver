"""
Utility Functions
Contains helper functions, API management, and general utilities
"""

import os
import pickle
from typing import Dict, Optional, Any
from pathlib import Path


class APIManager:
    """
    Manages API keys for various services.
    Supports both environment variables and explicit keys.
    """
    
    def __init__(self):
        self.api_keys = {}
        
    def set_api_key(self, service: str, api_key: str):
        """
        Set API key explicitly.
        
        Args:
            service: Service name (e.g., 'openai', 'anthropic')
            api_key: API key string
        """
        self.api_keys[service] = api_key
        
    def get_api_key(self, service: str, env_var: Optional[str] = None) -> Optional[str]:
        """
        Get API key from explicit setting or environment variable.
        
        Args:
            service: Service name
            env_var: Environment variable name (optional)
            
        Returns:
            API key string or None if not found
        """
        # Try explicit key first
        if service in self.api_keys:
            return self.api_keys[service]
        
        # Try environment variable
        if env_var:
            return os.getenv(env_var)
        
        # Try default environment variable pattern
        default_env = f"{service.upper()}_API_KEY"
        return os.getenv(default_env)
    
    def load_from_env(self, service: str, env_var: str):
        """
        Load API key from environment variable.
        
        Args:
            service: Service name
            env_var: Environment variable name
        """
        api_key = os.getenv(env_var)
        if api_key:
            self.api_keys[service] = api_key
        else:
            raise ValueError(f"Environment variable {env_var} not found")


def save_results(results: Dict, filepath: str):
    """
    Save results to pickle file.
    
    Args:
        results: Dictionary of results
        filepath: File path to save
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from pickle file.
    
    Args:
        filepath: File path to load
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"✓ Results loaded from {filepath}")
    return results


def print_summary(results: Dict, title: str = "TEXT METRICS SUMMARY"):
    """
    Print a formatted summary of metrics.
    
    Args:
        results: Dictionary of computed metrics
        title: Summary title
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    # Basic stats
    if 'basic_stats' in results:
        print(f"\n📊 Basic Statistics:")
        stats = results['basic_stats']
        if 'num_sentences' in stats:
            print(f"  Sentences: {stats['num_sentences']}")
        if 'num_tokens' in stats:
            print(f"  Tokens: {stats['num_tokens']}")
        if 'text_length' in results:
            print(f"  Characters: {results['text_length']}")
    
    # Shannon entropy
    if 'shannon_entropy' in results:
        print(f"\n📈 Shannon Entropy:")
        entropy = results['shannon_entropy']
        print(f"  Entropy: {entropy['entropy']:.3f}")
        print(f"  Normalized: {entropy['normalized_entropy']:.3f}")
    
    # POS features
    if 'pos_features' in results:
        print(f"\n🔤 POS Features:")
        pos = results['pos_features']
        print(f"  Lexical Ratio: {pos['lexical_ratio']:.3f}")
        print(f"  Noun Ratio: {pos['noun_ratio']:.3f}")
        print(f"  Verb Ratio: {pos['verb_ratio']:.3f}")
    
    # Lexical richness
    if 'lexical_richness' in results:
        print(f"\n📚 Lexical Richness:")
        lr = results['lexical_richness']
        print(f"  TTR: {lr['ttr']:.3f}")
        print(f"  RTTR: {lr['rttr']:.3f}")
    
    # Entity specificity
    if 'entity_specificity' in results:
        print(f"\n👤 Entity Specificity:")
        es = results['entity_specificity']
        print(f"  Specificity: {es['entity_specificity']:.3f}")
        print(f"  Unique Entities: {es['unique_entities']}")
    
    # Embedding variance
    if 'embedding_variance' in results:
        print(f"\n🔢 Embedding Variance:")
        print(f"  Variance: {results['embedding_variance']:.3f}")
    
    # Sentiment
    if 'sentiment_variance' in results:
        print(f"\n😊 Sentiment Metrics:")
        sv = results['sentiment_variance']
        print(f"  Mean: {sv['sentiment_mean']:.3f}")
        print(f"  Variance: {sv['sentiment_variance']:.3f}")
    
    print("\n" + "="*60 + "\n")


def extract_sentences(text: str) -> list:
    """
    Simple sentence extraction (for when NLP pipeline not available).
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return sentences


def count_tokens(text: str) -> int:
    """
    Simple token counting (for when NLP pipeline not available).
    
    Args:
        text: Input text
        
    Returns:
        Approximate token count
    """
    return len([w for w in text.split() if w.strip()])
