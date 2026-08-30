import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA = os.path.join(DATA_DIR, "raw", "train.csv")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

# Model
MODEL_NAME = "sagorsarker/bangla-bert-base"  # or "xlm-roberta-base"

# Label spaces (must match dataset generation)
SUBJECTS = ["Biology", "Physics", "Chemistry", "Mathematics", "ICT", "Bangla", "English"]
TOPICS = ["Photosynthesis", "Cell Division", "Genetics", "Ecology", "Human Body", 
          "Motion", "Force", "Energy", "Optics", "Electricity",
          "Organic Chemistry", "Acids & Bases", "Periodic Table", "Chemical Bonds",
          "Algebra", "Geometry", "Calculus", "Statistics",
          "Networking", "Programming", "Database", "Cybersecurity",
          "Grammar", "Literature", "Poetry",
          "Vocabulary", "Comprehension"]
Q_TYPES = ["MCQ", "Short Answer", "Creative", "Descriptive", "Numerical"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
COGNITIVE = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]

# Mapping dicts for label -> id
LABEL_MAPS = {
    "subject": {v: i for i, v in enumerate(SUBJECTS)},
    "topic": {v: i for i, v in enumerate(TOPICS)},
    "question_type": {v: i for i, v in enumerate(Q_TYPES)},
    "difficulty": {v: i for i, v in enumerate(DIFFICULTIES)},
    "cognitive_level": {v: i for i, v in enumerate(COGNITIVE)},
}

# Inverse mappings for prediction
INV_LABEL_MAPS = {
    "subject": {i: v for v, i in LABEL_MAPS["subject"].items()},
    "topic": {i: v for v, i in LABEL_MAPS["topic"].items()},
    "question_type": {i: v for v, i in LABEL_MAPS["question_type"].items()},
    "difficulty": {i: v for v, i in LABEL_MAPS["difficulty"].items()},
    "cognitive_level": {i: v for v, i in LABEL_MAPS["cognitive_level"].items()},
}

# Number of classes per task
NUM_CLASSES = {
    "subject": len(SUBJECTS),
    "topic": len(TOPICS),
    "question_type": len(Q_TYPES),
    "difficulty": len(DIFFICULTIES),
    "cognitive_level": len(COGNITIVE),
}
