# 🇧🇩 BanglaEduAI

**Multi-Task Transformer Framework for Bangla Educational Question Understanding**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

## 📌 Overview
BanglaEduAI is an NLP system that automatically analyzes a Bangla educational question and extracts:
- **Subject** (e.g., Biology, Physics)
- **Topic** (e.g., Photosynthesis, Motion)
- **Question Type** (MCQ, Creative, etc.)
- **Difficulty** (Easy, Medium, Hard)
- **Cognitive Level** (Based on Bloom's Taxonomy)

It uses a shared BanglaBERT encoder with multi-task classification heads, trained on a synthetic dataset.

## 🏗️ Architecture


Question → Preprocess → BanglaBERT → [Subject, Topic, Type, Difficulty, Cognitive]



## 🚀 Quick Start

### 1. Install dependencies
bash
pip install -r requirements.txt

python data/raw/generate_data.py

python src/training/train.py


uvicorn api.main:app --reload


python frontend/app.py





Input:

  "আলোক সংশ্লেষণ প্রক্রিয়ায় ক্লোরোফিলের ভূমিকা ব্যাখ্যা কর।"
  
Output:
{
  "subject": {"label": "Biology", "confidence": 0.96},
  "topic": {"label": "Photosynthesis", "confidence": 0.91},
  "question_type": {"label": "Creative", "confidence": 0.94},
  "difficulty": {"label": "Medium", "confidence": 0.83},
  "cognitive_level": {"label": "Understanding", "confidence": 0.88}
}
