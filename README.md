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
