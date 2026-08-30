import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import numpy as np

from src.config import MODEL_DIR, MODEL_NAME, INV_LABEL_MAPS, NUM_CLASSES
from src.models.multitask_model import BanglaEduMultiTaskModel
from src.preprocessing.cleaner import clean_bangla_text

app = FastAPI(title="BanglaEduAI API")

# ---------- Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# We need to load the multi-task model
model = BanglaEduMultiTaskModel(MODEL_NAME, NUM_CLASSES)
model_path = os.path.join(MODEL_DIR, "best_model.pth")
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}. Please run training first.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---------- Request Schema ----------
class QuestionRequest(BaseModel):
    question: str

# ---------- Inference ----------
def predict(text: str):
    cleaned = clean_bangla_text(text)
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    result = {}
    for task, logits in outputs["logits"].items():
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = np.argmax(probs)
        result[task] = {
            "label": INV_LABEL_MAPS[task][pred_id],
            "confidence": float(probs[pred_id])
        }
    return result

# ---------- Routes ----------
@app.post("/api/v1/analyze")
async def analyze_question(request: QuestionRequest):
    try:
        result = predict(request.question)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "BanglaEduAI API is running. Use POST /api/v1/analyze"}
