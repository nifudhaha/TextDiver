"""
ctrLLM - Controversial Topic Representation in LLM

A comprehensive toolkit for analyzing text diversity, bias, and 
controversial topic representation in language models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Import main classes for easy access
from .metrics import TextMetrics
from .syntactic import SyntacticAnalyzer
from .semantic import SemanticAnalyzer
from .entity import EntityAnalyzer
from .sentiment import SentimentAnalyzer
from .argument import ArgumentAnalyzer
from .rule_based import RuleBasedAnalyzer
from .utils import (
    APIManager,
    save_results,
    load_results,
    print_summary
)

__all__ = [
    "TextMetrics",
    "SyntacticAnalyzer",
    "SemanticAnalyzer", 
    "EntityAnalyzer",
    "SentimentAnalyzer",
    "ArgumentAnalyzer",
    "RuleBasedAnalyzer",
    "APIManager",
    "save_results",
    "load_results",
    "print_summary"
]