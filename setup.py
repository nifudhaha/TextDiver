"""
Setup script for ctrLLM package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README with utf-8 encoding
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"

if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8", errors="ignore")
else:
    long_description = ""

setup(
    name="ctrllm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Controversial Topic Representation in LLM - A toolkit for analyzing text diversity and bias",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ctrllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "stanza>=1.4.0",
        "sentence-transformers>=2.0.0",
        "transformers>=4.0.0",
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
    keywords='llm controversial-topics text-analysis bias-detection diversity-metrics nlp',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/ctrllm/issues',
        'Source': 'https://github.com/yourusername/ctrllm',
        'Documentation': 'https://github.com/yourusername/ctrllm#readme',
    },
)
