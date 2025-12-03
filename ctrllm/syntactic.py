"""
Syntactic Analysis Module
Contains metrics related to syntax, grammar, and lexical features
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import stanza


class SyntacticAnalyzer:
    """
    Analyzes syntactic and lexical properties of text including:
    - POS features
    - Vocabulary complexity
    - Lexical richness
    - Shannon's entropy
    """
    
    def __init__(self, lang: str = "en"):
        """
        Initialize syntactic analyzer with Stanza.
        
        Args:
            lang: Language code (e.g., 'en', 'zh', 'es')
        """
        try:
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma',
                download_method=None
            )
        except Exception:
            print(f"Downloading Stanza {lang} models...")
            stanza.download(lang)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma'
            )
        
        self.doc = None
        self.sentences = []
        
    def process_text(self, text: str):
        """
        Process text with Stanza.
        
        Args:
            text: Input text to analyze
        """
        self.doc = self.nlp(text)
        self.sentences = [sent.text.strip() for sent in self.doc.sentences 
                         if sent.text.strip()]
    
    def shannon_entropy(self, use_lemma: bool = True) -> Dict[str, float]:
        """
        Calculate Shannon's entropy of token/lemma distribution.
        Formula: H = -Σ p(w) log p(w)
        
        Args:
            use_lemma: If True, use lemmas; if False, use raw tokens
            
        Returns:
            Dictionary with entropy metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        # Extract tokens/lemmas
        tokens = []
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM', 'X']:
                    if use_lemma:
                        tokens.append(word.lemma.lower())
                    else:
                        tokens.append(word.text.lower())
        
        if not tokens:
            return {
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'unique_tokens': 0,
                'total_tokens': 0
            }
        
        # Calculate probability distribution
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        probs = np.array([count / total_tokens for count in token_counts.values()])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalized entropy
        unique_tokens = len(token_counts)
        normalized_entropy = entropy / np.log2(unique_tokens) if unique_tokens > 1 else 0.0
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens
        }
    
    def pos_features(self) -> Dict[str, float]:
        """
        Calculate POS distribution features.
        
        Returns:
            Dictionary with POS ratios and variability
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        # Collect words
        words = []
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM']:
                    words.append(word)
        
        total_tokens = len(words)
        
        if total_tokens == 0:
            return {
                'noun_ratio': 0.0,
                'verb_ratio': 0.0,
                'adj_ratio': 0.0,
                'adv_ratio': 0.0,
                'lexical_ratio': 0.0,
                'pos_variability': 0.0,
                'distinct_pos_tags': 0
            }
        
        # Count POS tags
        pos_counts = Counter([word.upos for word in words])
        
        noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        verb_count = pos_counts.get('VERB', 0)
        adj_count = pos_counts.get('ADJ', 0)
        adv_count = pos_counts.get('ADV', 0)
        
        lexical_count = noun_count + verb_count + adj_count + adv_count
        pos_variability = len(pos_counts) / total_tokens
        
        return {
            'noun_ratio': noun_count / total_tokens,
            'verb_ratio': verb_count / total_tokens,
            'adj_ratio': adj_count / total_tokens,
            'adv_ratio': adv_count / total_tokens,
            'lexical_ratio': lexical_count / total_tokens,
            'pos_variability': pos_variability,
            'distinct_pos_tags': len(pos_counts)
        }
    
    def vocabulary_complexity(self) -> Dict[str, float]:
        """
        Calculate vocabulary complexity (POS tag diversity per lemma).
        Formula: Vocab_complexity = (1/#lemmas) Σ k_ℓ
        
        Returns:
            Dictionary with complexity metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        lemma_pos = {}
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM']:
                    lemma = word.lemma.lower()
                    if lemma not in lemma_pos:
                        lemma_pos[lemma] = set()
                    lemma_pos[lemma].add(word.upos)
        
        if not lemma_pos:
            return {
                'vocab_complexity': 0.0,
                'unique_lemmas': 0,
                'avg_pos_per_lemma': 0.0,
                'max_pos_per_lemma': 0,
                'lemmas_with_multiple_pos': 0
            }
        
        pos_counts = [len(pos_set) for pos_set in lemma_pos.values()]
        vocab_complexity = np.mean(pos_counts)
        
        return {
            'vocab_complexity': float(vocab_complexity),
            'unique_lemmas': len(lemma_pos),
            'avg_pos_per_lemma': float(vocab_complexity),
            'max_pos_per_lemma': max(pos_counts),
            'lemmas_with_multiple_pos': sum(1 for c in pos_counts if c > 1)
        }
    
    def lexical_richness(self) -> Dict[str, float]:
        """
        Calculate lexical richness metrics (TTR, RTTR, CTTR).
        
        Returns:
            Dictionary with richness metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        tokens = []
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM', 'X'] and word.text.isalpha():
                    tokens.append(word.text.lower())
        
        if not tokens:
            return {
                'ttr': 0.0,
                'rttr': 0.0,
                'cttr': 0.0,
                'unique_tokens': 0,
                'total_tokens': 0
            }
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        ttr = unique_tokens / total_tokens
        rttr = unique_tokens / np.sqrt(total_tokens)
        cttr = unique_tokens / np.sqrt(2 * total_tokens)
        
        return {
            'ttr': float(ttr),
            'rttr': float(rttr),
            'cttr': float(cttr),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens
        }
    
    def get_all_metrics(self) -> Dict:
        """
        Compute all syntactic metrics.
        
        Returns:
            Dictionary with all syntactic metrics
        """
        return {
            'shannon_entropy': self.shannon_entropy(),
            'pos_features': self.pos_features(),
            'vocabulary_complexity': self.vocabulary_complexity(),
            'lexical_richness': self.lexical_richness()
        }
