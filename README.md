# ctrLLM

**Controversial Topic Representation in LLM**

A comprehensive Python package for analyzing text diversity, bias, and controversial topic representation in language models.

---

## 🎯 Purpose

This toolkit helps researchers and developers analyze how different sources (Wikipedia, Britannica, LLMs) represent controversial topics by measuring:

- **Text diversity** (vocabulary, semantic, lexical)
- **Perspective multiplexity** (multiple viewpoints)
- **Bias detection** (single vs. multi-perspective)
- **Content quality** (complexity, richness)

---

## 🚀 Quick Start

### Installation

```bash
# Install package
pip install -e .

# Install dependencies (including OpenAI)
pip install -r requirements.txt

# Download Stanza models
python -c "import stanza; stanza.download('en')"

# Set OpenAI API key for argument detection
export OPENAI_API_KEY='your-openai-api-key'
```

### Basic Usage

```python
from ctrllm import TextMetrics, print_summary

# Initialize
metrics = TextMetrics(lang='en')

# Analyze text
text = "Your controversial topic text here..."
results = metrics.compute_all(text)

# Print results
print_summary(results)
```

---

## 📦 Package Structure

```
ctrllm_package/
├── ctrllm/                    # Main package
│   ├── __init__.py           # Package initialization
│   ├── metrics.py            # Main interface
│   ├── syntactic.py          # Syntactic analysis
│   ├── semantic.py           # Semantic analysis
│   ├── entity.py             # Entity analysis
│   ├── sentiment.py          # Sentiment analysis
│   └── utils.py              # Utilities
├── demo.py                    # Demo script
├── test.py                    # Test suite
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

---

## 📊 Features

### Syntactic Analysis (`syntactic.py`)
- **Shannon Entropy** - Vocabulary distribution
- **POS Features** - Grammatical analysis
- **Vocabulary Complexity** - Word flexibility
- **Lexical Richness** - TTR, RTTR, CTTR

### Semantic Analysis (`semantic.py`)
- **Embedding Variance** - Semantic diversity
- **Pairwise Similarity** - Sentence similarity

### Entity Analysis (`entity.py`)
- **Entity Specificity** - Named entity density
- **Entity Distribution** - Person/Org/Location counts

### Sentiment Analysis (`sentiment.py`)
- **Sentiment Variance** - Emotional variability
- **Sentiment Distribution** - Positive/negative/neutral ratios

### **Argument Analysis (`argument.py`)** ⭐ NEW
- **Main vs. Fringe Perspective** - Main/fringe viewpoint ratio
- **Argument Diversity** - Argument topic diversity  
- **Argument Distinctness** - Cluster separation
- **Argumentativeness** - Argument density

**Uses LLM-based detection for high accuracy** (requires OpenAI API)

### Utilities (`utils.py`)
- **API Management** - Environment variables + explicit keys
- **Save/Load** - Pickle serialization
- **Print Summary** - Formatted output

---

## 💻 Usage Examples

### Example 1: Full Analysis

```python
from ctrllm import TextMetrics

metrics = TextMetrics(lang='en')
results = metrics.compute_all(text)

# Access specific metrics
print(results['shannon_entropy']['entropy'])
print(results['embedding_variance'])
print(results['entity_specificity'])
```

### Example 2: Individual Analyzers

```python
from ctrllm import SyntacticAnalyzer, SemanticAnalyzer, ArgumentAnalyzer

# Use individual classes
syntactic = SyntacticAnalyzer(lang='en')
syntactic.process_text(text)
entropy = syntactic.shannon_entropy()

semantic = SemanticAnalyzer()
semantic.set_sentences(sentences)
variance = semantic.embedding_variance()

# Argument analysis (LLM-based, requires OPENAI_API_KEY)
argument = ArgumentAnalyzer(
    lang='en',
    llm_model='gpt-4o-mini'
)
argument.process_text(text)
arg_results = argument.get_all_metrics(batch_delay=0.05)

print(f"Main ratio: {arg_results['main_vs_fringe']['main_ratio']:.3f}")
print(f"Diversity: {arg_results['argument_diversity']['arg_diversity']:.3f}")
```

---

## 📝 Running Demo

```bash
python demo.py
```

This will:
1. Analyze a controversial topic text
2. Show syntactic-only analysis
3. Compare two texts (simple vs. complex)

---

## 🎓 Use Cases

1. **Compare information sources** (Wikipedia vs. Britannica vs. LLM)
2. **Detect bias** in controversial topic coverage
3. **Measure perspective diversity** in content
4. **Assess text quality** and complexity
5. **Academic research** on LLM representations

---

## 📚 Dependencies

```
stanza                  # NLP processing
sentence-transformers   # Sentence embeddings
transformers           # Sentiment analysis
torch                  # PyTorch backend
numpy                  # Numerical operations
scikit-learn          # ML utilities
```

---

## 🔧 API Reference

### Main Class: `TextMetrics`

```python
TextMetrics(lang='en', embedding_model='...', sentiment_model='...')

# Methods
.compute_all(text, **options)      # All metrics
.compute_syntactic(text)            # Syntactic only
.compute_semantic(text)             # Semantic only
.compute_entity(text)               # Entity only
.compute_sentiment(text)            # Sentiment only
```

### Individual Analyzers

See source code docstrings for detailed API documentation.

---

## 📄 License

MIT License - Free for academic and commercial use

---


