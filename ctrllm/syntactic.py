"""
Syntactic Analysis Module
Contains metrics related to syntax, grammar, and lexical features
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import stanza
from lexicalrichness import LexicalRichness
from nltk.corpus import wordnet as wn
import nltk


class SyntacticAnalyzer:
    """
    Analyzes syntactic and lexical properties of text including:
    - POS features
    - Vocabulary complexity
    - Lexical richness
    - Shannon's entropy
    - Polysemy (semantic ambiguity)
    """
    
    def __init__(self, lang: str = "en"):
        """
        Initialize syntactic analyzer with Stanza and WordNet.
        
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
        
        # Download WordNet if not available
        try:
            wn.ensure_loaded()
        except:
            print("Downloading WordNet...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')  # Open Multilingual WordNet
        
        self.doc = None
        self.sentences = []
        self.text = None
        self.lang = lang
        
    def process_text(self, text: str):
        """
        Process text with Stanza.
        
        Args:
            text: Input text to analyze
        """
        self.text = text
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
    
    def polysemy(self) -> Dict[str, float]:
        """
        Calculate polysemy using WordNet synsets.
        Formula: Polysemy = (1/#lemmas) Σ synset_count(ℓ)
        
        Polysemy measures semantic ambiguity - how many different meanings
        each word can have on average.
        
        Returns:
            Dictionary with polysemy metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        # Map Stanza POS to WordNet POS
        pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'PROPN': wn.NOUN  # Proper nouns treated as nouns
        }
        
        lemma_synsets = []
        lemma_details = []
        
        for sent in self.doc.sentences:
            for word in sent.words:
                # Only consider content words with WordNet POS
                if word.upos in pos_map:
                    lemma = word.lemma.lower()
                    wn_pos = pos_map[word.upos]
                    
                    # Get synsets for this lemma with its POS
                    synsets = wn.synsets(lemma, pos=wn_pos)
                    synset_count = len(synsets)
                    
                    lemma_synsets.append(synset_count)
                    lemma_details.append({
                        'lemma': lemma,
                        'pos': word.upos,
                        'synset_count': synset_count
                    })
        
        if not lemma_synsets:
            return {
                'polysemy': 0.0,
                'avg_synsets_per_lemma': 0.0,
                'max_synsets': 0,
                'min_synsets': 0,
                'std_synsets': 0.0,
                'monosemous_ratio': 0.0,  
                'polysemous_ratio': 0.0,  
                'high_polysemy_ratio': 0.0,  
                'total_lemmas': 0,
                'lemmas_with_synsets': 0
            }
        
        # Calculate statistics
        polysemy_score = np.mean(lemma_synsets)
        synsets_array = np.array(lemma_synsets)
        
        # Count different categories
        monosemous = np.sum(synsets_array == 1)
        polysemous = np.sum(synsets_array > 1)
        high_polysemy = np.sum(synsets_array > 5)
        with_synsets = np.sum(synsets_array > 0)
        total = len(lemma_synsets)
        
        return {
            'polysemy': float(polysemy_score),
            'avg_synsets_per_lemma': float(polysemy_score),
            'max_synsets': int(np.max(synsets_array)),
            'min_synsets': int(np.min(synsets_array)),
            'std_synsets': float(np.std(synsets_array)),
            'monosemous_ratio': monosemous / total,  
            'polysemous_ratio': polysemous / total,  
            'high_polysemy_ratio': high_polysemy / total,  
            'total_lemmas': total,
            'lemmas_with_synsets': int(with_synsets)
        }
    
    def lexical_richness(self) -> Dict[str, float]:
        """
        Calculate lexical richness metrics using lexical_richness library.
        Includes: TTR, RTTR, CTTR, MSTTR, MATTR, HD-D, MTLD, and more.
        
        Returns:
            Dictionary with comprehensive richness metrics
        """
        if self.text is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        try:
            # Initialize LexicalRichness with the text
            lex = LexicalRichness(self.text)
            
            # Calculate various metrics
            metrics = {
                # Basic metrics
                'ttr': lex.ttr,  # Type-Token Ratio
                'rttr': lex.rttr,  # Root TTR
                'cttr': lex.cttr,  # Corrected TTR
                'Herdan': lex.Herdan,  # Herdan's C
                'Summer': lex.Summer,  # Summer's S
                'Dugast': lex.Dugast,  # Dugast's U
                'Maas': lex.Maas,  # Maas's a²
                
                # Advanced metrics
                'msttr': lex.msttr(segment_window=100),  # Mean-Segmental TTR
                'mattr': lex.mattr(window_size=100),  # Moving-Average TTR
                'hdd': lex.hdd(draws=42),  # HD-D (vocd-D)
                'mtld': lex.mtld(threshold=0.72),  # MTLD
                
                # Word statistics
                'terms': lex.terms,  # Unique words
                'words': lex.words,  # Total words
            }
            
            return {k: float(v) if v is not None else 0.0 for k, v in metrics.items()}
            
        except Exception as e:
            # Fallback to basic metrics if library fails
            print(f"Warning: lexical_richness library failed ({e}). Using basic metrics.")
            
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
                    'terms': 0,
                    'words': 0
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
                'terms': unique_tokens,
                'words': total_tokens
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
            'polysemy': self.polysemy(),
            'lexical_richness': self.lexical_richness()
        }
