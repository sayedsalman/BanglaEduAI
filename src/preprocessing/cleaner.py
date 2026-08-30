import re

def clean_bangla_text(text: str) -> str:
    """Remove extra spaces, normalize punctuation, strip."""
    # Remove URLs, emojis, special symbols (keep Bangla, English, digits, basic punctuation)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s\.\?\!\,\;\-\:]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
