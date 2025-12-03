"""
Rule-Based Analysis Module
Contains metrics that require predefined rules or patterns, 
typically implemented using LLM for complex semantic understanding.
"""

import os
import json
import time
from typing import List, Dict, Optional
from collections import Counter

# OpenAI for LLM-based analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RuleBasedAnalyzer:
    """
    Analyzes text using rule-based metrics with LLM assistance:
    
    1. Balanced Pro/Con - Detects symmetric balanced sentences using LLM
    2. Narrative Roles - Identifies Hero/Villain/Victim using LLM
    3. Harm Index - Quantifies harm using LLM
    4. Devil-Angel Shift - Detects role reversals (based on narrative roles)
    
    All metrics except devil_angel_shift require LLM.
    """
    
    def __init__(self,
                 llm_model: str = "gpt-4o-mini",
                 llm_temperature: float = 0.0):
        """
        Initialize rule-based analyzer.
        
        Args:
            llm_model: OpenAI model name
            llm_temperature: Temperature for LLM (0.0 for deterministic)
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
        
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_client = OpenAI()
        self.llm_cache: Dict[str, Dict] = {}
        
        print(f"RuleBasedAnalyzer initialized with LLM model: {llm_model}")
        
        # Data storage
        self.text = ""
        self.topic = ""
        self.arguments = []
        self.stances = []
        self.narrative_roles = []
        self.harm_data = []
    
    def set_text(self, text: str, topic: str = ""):
        """
        Set the text to analyze.
        
        Args:
            text: Input text
            topic: Optional topic/subject being discussed
        """
        self.text = text
        self.topic = topic
    
    def set_arguments(self, arguments: List[str]):
        """
        Set pre-detected arguments for stance analysis.
        
        Note: This method is kept for API compatibility but is no longer 
        required since balanced_pro_con now uses connective-based detection.
        
        Args:
            arguments: List of argument sentences (not used)
        """
        self.arguments = arguments
    
    # ==================== Balanced Pro and Con ====================
    
    def balanced_pro_con(self, batch_delay: float = 0.05) -> Dict:
        """
        Detect balanced presentation using LLM to identify symmetric pro-con sentences.
        
        Method:
        Uses LLM to identify sentences that present both sides, such as:
        - "Some argue that X, while others believe that Y"
        - "On one hand A, on the other hand B"
        - "Proponents claim X, whereas opponents say Y"
        
        Recognition criteria:
        1. Sentence contains contrasting viewpoints
        2. Both viewpoints are presented within the same sentence or closely connected sentences
        3. Uses symmetric patterns or contrastive connectives
        
        Formula:
        - Balanced_ratio = #BalancedSent / N
        - Where N = total number of sentences
        
        Args:
            batch_delay: Delay between API calls
            
        Returns:
            {
                'balanced_ratio': float [0, 1],
                'num_balanced_sentences': int,
                'total_sentences': int,
                'balanced_sentences': List[Dict],
                'interpretation': str
            }
        """
        if not self.text:
            return {
                'balanced_ratio': 0.0,
                'num_balanced_sentences': 0,
                'total_sentences': 0,
                'balanced_sentences': [],
                'interpretation': 'No text provided'
            }
        
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
        
        total_sentences = len(sentences)
        balanced_sentences = []
        
        # Check each sentence with LLM
        for sent in sentences:
            # Check cache
            cache_key = f"balanced:{sent[:100]}"
            if cache_key in self.llm_cache:
                result = self.llm_cache[cache_key]
            else:
                result = self._check_balanced_sentence(sent, batch_delay)
                self.llm_cache[cache_key] = result
            
            if result['is_balanced']:
                balanced_sentences.append({
                    'sentence': sent,
                    'confidence': result['confidence'],
                    'pattern_type': result.get('pattern_type', 'unknown')
                })
        
        # Calculate ratio
        num_balanced = len(balanced_sentences)
        balanced_ratio = num_balanced / total_sentences if total_sentences > 0 else 0.0
        
        return {
            'balanced_ratio': float(balanced_ratio),
            'num_balanced_sentences': num_balanced,
            'total_sentences': total_sentences,
            'balanced_sentences': balanced_sentences,
        }
    
    def _check_balanced_sentence(self, sentence: str, batch_delay: float) -> Dict:
        """
        Use LLM to check if a sentence presents both pro and con viewpoints.
        
        Args:
            sentence: The sentence to check
            batch_delay: Delay after API call
            
        Returns:
            {
                'is_balanced': bool,
                'confidence': float,
                'explanation': str,
                'pattern_type': str
            }
        """
        prompt = f"""Analyze if this sentence presents BOTH pro and con viewpoints (balanced presentation):

Sentence: "{sentence}"

A sentence is "balanced" if it:
1. Presents two contrasting viewpoints within the same sentence
2. Uses symmetric patterns like "Some...others", "On one hand...on the other hand", "Proponents...opponents"
3. OR uses contrastive connectives (while, whereas, however, although, but) to connect opposing views

Examples of BALANCED sentences:
- "Some argue that X is good, while others believe Y is better."
- "On one hand, A is beneficial; on the other hand, B has drawbacks."
- "Proponents claim X works, whereas opponents say Y fails."
- "Although A has advantages, B presents significant challenges."

Examples of NOT BALANCED sentences:
- "Climate change is a serious threat." (only one viewpoint)
- "Scientists agree that action is needed." (only one viewpoint)
- "However, some disagree." (only states disagreement, no specific opposing view)

Output STRICT JSON:
{{
  "is_balanced": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of why it is or isn't balanced",
  "pattern_type": "symmetric_pattern" | "contrastive_connective" | "none"
}}"""
        
        # Call LLM with retries
        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in discourse analysis and argumentation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    response_format={"type": "json_object"}
                )
                
                raw = response.choices[0].message.content.strip()
                result = json.loads(raw)
                
                balanced_dict = {
                    'is_balanced': result.get('is_balanced', False),
                    'pattern_type': result.get('pattern_type', 'none')
                }
                
                time.sleep(batch_delay)
                return balanced_dict
                
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to check balanced sentence: {e}")
                    return {
                        'is_balanced': False,
                        'confidence': 0.0,
                        'explanation': f'Error: {str(e)}',
                        'pattern_type': 'error'
                    }
                time.sleep(1 * (attempt + 1))
    
    def _interpret_balanced_ratio(self, ratio: float) -> str:
        """Interpret balanced ratio."""
        if ratio >= 0.5:
            return "Highly balanced - many sentences present both sides"
        elif ratio >= 0.3:
            return "Moderately balanced - some two-sided presentation"
        elif ratio >= 0.1:
            return "Low balance - few balanced sentences"
        else:
            return "Very low balance - minimal two-sided presentation"
    
    
    # ==================== Narrative Roles ====================
    
    def detect_narrative_roles(self,
                              batch_delay: float = 0.05,
                              max_retries: int = 3) -> Dict:
        """
        Detect narrative roles (Hero/Villain/Victim) in the text.
        
        Args:
            batch_delay: Delay between API calls
            max_retries: Maximum retries
            
        Returns:
            {
                'heroes': List[str],
                'villains': List[str],
                'victims': List[str],
                'hero_count': int,
                'villain_count': int,
                'victim_count': int,
                'interpretation': str
            }
        """
        # Check cache
        cache_key = f"roles:{self.text[:100]}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        prompt = f"""Analyze the narrative roles in this text:

Text: "{self.text}"

Identify entities playing these roles:
- Hero: Portrayed positively, as protagonist, savior, or doing good
- Villain: Portrayed negatively, as antagonist, causing harm, or doing wrong
- Victim: Portrayed as suffering, harmed, or disadvantaged

Output STRICT JSON:
{{
  "heroes": ["entity1", "entity2", ...],
  "villains": ["entity1", "entity2", ...],
  "victims": ["entity1", "entity2", ...],
  "explanation": "brief justification"
}}

Note: An entity can play multiple roles in different contexts."""
        
        # Call LLM with retries
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a narrative analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    response_format={"type": "json_object"}
                )
                
                raw = response.choices[0].message.content.strip()
                result = json.loads(raw)
                
                heroes = result.get('heroes', [])
                villains = result.get('villains', [])
                victims = result.get('victims', [])
                
                roles_dict = {
                    'heroes': heroes,
                    'villains': villains,
                    'victims': victims,
                    'hero_count': len(heroes),
                    'villain_count': len(villains),
                    'victim_count': len(victims),
                }
                
                # Cache result
                self.llm_cache[cache_key] = roles_dict
                self.narrative_roles = roles_dict
                
                time.sleep(batch_delay)
                return roles_dict
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to detect narrative roles: {e}")
                    return {
                        'heroes': [],
                        'villains': [],
                        'victims': [],
                        'hero_count': 0,
                        'villain_count': 0,
                        'victim_count': 0,
                        'interpretation': 'Analysis failed'
                    }
                time.sleep(1 * (attempt + 1))
    
    def _interpret_roles(self, heroes: List, villains: List, victims: List) -> str:
        """Interpret narrative roles distribution."""
        total = len(heroes) + len(villains) + len(victims)
        if total == 0:
            return "No clear narrative roles"
        
        if len(villains) > 0 and len(heroes) > 0:
            return f"Clear narrative structure with {len(heroes)} hero(es) vs {len(villains)} villain(s)"
        elif len(villains) > 0:
            return f"Villain-focused narrative with {len(villains)} antagonist(s)"
        elif len(heroes) > 0:
            return f"Hero-focused narrative with {len(heroes)} protagonist(s)"
        else:
            return f"Victim-centered narrative with {len(victims)} victim(s)"
    
    # ==================== Harm Metrics ====================
    
    def detect_harm_index(self,
                         batch_delay: float = 0.05,
                         max_retries: int = 3) -> Dict:
        """
        Detect harm mentioned in text (number of people affected).
        
        Args:
            batch_delay: Delay between API calls
            max_retries: Maximum retries
            
        Returns:
            {
                'total_harm': int,
                'harm_mentions': List[Dict],
                'harm_categories': Dict[str, int],
                'interpretation': str
            }
        """
        # Check cache
        cache_key = f"harm:{self.text[:100]}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        prompt = f"""Analyze harm or negative impacts mentioned in this text:

Text: "{self.text}"

Extract:
1. Number of people affected (deaths, injuries, displaced, etc.)
2. Type of harm (death, injury, economic loss, environmental damage, etc.)
3. Specific numbers or estimates mentioned

Output STRICT JSON:
{{
  "harm_mentions": [
    {{
      "type": "death|injury|displacement|economic|environmental|other",
      "count": number or -1 if unknown,
      "description": "brief description"
    }}
  ],
  "total_affected": total number or -1 if cannot determine,
  "explanation": "summary of harm mentioned"
}}"""
        
        # Call LLM with retries
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a harm quantification expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    response_format={"type": "json_object"}
                )
                
                raw = response.choices[0].message.content.strip()
                result = json.loads(raw)
                
                harm_mentions = result.get('harm_mentions', [])
                total_affected = result.get('total_affected', 0)
                
                # Categorize harm
                harm_categories = {}
                for mention in harm_mentions:
                    harm_type = mention.get('type', 'other')
                    count = mention.get('count', 0)
                    if count > 0:
                        harm_categories[harm_type] = harm_categories.get(harm_type, 0) + count
                
                harm_dict = {
                    'total_harm': int(total_affected) if total_affected > 0 else 0,
                    'harm_mentions': harm_mentions,
                    'harm_categories': harm_categories,
                    'num_harm_types': len(harm_categories),
                }
                
                # Cache result
                self.llm_cache[cache_key] = harm_dict
                self.harm_data = harm_dict
                
                time.sleep(batch_delay)
                return harm_dict
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to detect harm: {e}")
                    return {
                        'total_harm': 0,
                        'harm_mentions': [],
                        'harm_categories': {},
                        'num_harm_types': 0,
                        'interpretation': 'Analysis failed'
                    }
                time.sleep(1 * (attempt + 1))
    
    def _interpret_harm(self, total: int, categories: Dict) -> str:
        """Interpret harm scale."""
        if total <= 0:
            return "No quantifiable harm mentioned"
        elif total < 100:
            return f"Small-scale harm: {total} people affected across {len(categories)} categories"
        elif total < 10000:
            return f"Medium-scale harm: {total} people affected across {len(categories)} categories"
        elif total < 1000000:
            return f"Large-scale harm: {total} people affected across {len(categories)} categories"
        else:
            return f"Massive-scale harm: {total}+ people affected across {len(categories)} categories"
    
    # ==================== Devil-Angel Shift ====================
    
    def detect_devil_angel_shift(self) -> Dict:
        """
        Calculate Devil-Angel Shift: difference between heroes and villains.
        
        Formula:
        DevilAngelShift = #Hero - #Villain
        
        Positive value: More heroes (angel-leaning narrative)
        Negative value: More villains (devil-leaning narrative)
        Zero: Balanced hero-villain ratio
        
        Requires narrative_roles to be detected first.
        
        Returns:
            {
                'devil_angel_shift': int,
                'num_heroes': int,
                'num_villains': int,
                'interpretation': str
            }
        """
        if not self.narrative_roles:
            print(" Narrative roles not detected yet, detecting now...")
            self.detect_narrative_roles()
        
        num_heroes = self.narrative_roles.get('hero_count', 0)
        num_villains = self.narrative_roles.get('villain_count', 0)
        
        shift = num_heroes - num_villains
        
        print(f"Devil-Angel Shift: {shift} ({num_heroes} heroes - {num_villains} villains)")
        
        return {
            'devil_angel_shift': shift,
            'num_heroes': num_heroes,
            'num_villains': num_villains,
            'interpretation': self._interpret_shift(shift)
        }
    
    def _interpret_shift(self, shift: int) -> str:
        """Interpret devil-angel shift value."""
        if shift > 0:
            return f"Angel-leaning narrative (+{shift}): More heroes than villains"
        elif shift < 0:
            return f"Devil-leaning narrative ({shift}): More villains than heroes"
        else:
            return "Balanced narrative: Equal heroes and villains"
    
    # ==================== Get All Metrics ====================
    
    def get_all_metrics(self, 
                       batch_delay: float = 0.05,
                       min_confidence: float = 0.5) -> Dict:
        """
        Compute all rule-based metrics.
        
        Args:
            batch_delay: Delay between API calls (for LLM-based metrics)
            min_confidence: Minimum confidence (reserved for future use)
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Balanced Pro and Con (LLM-based)
        results['balanced_pro_con'] = self.balanced_pro_con(
            batch_delay=batch_delay
        )
        
        # Narrative Roles (LLM-based)
        results['narrative_roles'] = self.detect_narrative_roles(
            batch_delay=batch_delay
        )
        
        # Harm Index (LLM-based)
        results['harm_index'] = self.detect_harm_index(
            batch_delay=batch_delay
        )
        
        # Devil-Angel Shift (requires narrative roles)
        results['devil_angel_shift'] = self.detect_devil_angel_shift()
        
        return results