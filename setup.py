from setuptools import setup, find_packages

setup(
    name="BanglaEduAI",
    version="1.0.0",
    description="Multi-Task Transformer Framework for Bangla Educational Question Understanding",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(include=["src", "src.*", "api", "frontend"]),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "gradio>=4.0.0",
        "python-multipart",
        "sentencepiece",
        "protobuf",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "banglaeduai-train=src.training.train:main",
            "banglaeduai-api=api.main:app",
        ]
    },
)
