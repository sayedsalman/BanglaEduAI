import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from config import (
    RAW_DATA, MODEL_DIR, MODEL_NAME, LABEL_MAPS, NUM_CLASSES
)
from preprocessing.cleaner import clean_bangla_text
from models.multitask_model import BanglaEduMultiTaskModel

# ---------- Dataset Class ----------
class EduQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = ["subject", "topic", "question_type", "difficulty", "cognitive_level"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = clean_bangla_text(row["question"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        # Labels
        for col in self.label_cols:
            item[col] = torch.tensor(LABEL_MAPS[col][row[col]], dtype=torch.long)
        return item

# ---------- Collate Function ----------
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = {
        col: torch.stack([item[col] for item in batch])
        for col in ["subject", "topic", "question_type", "difficulty", "cognitive_level"]
    }
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------- Training Loop ----------
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = {k: 0 for k in LABEL_MAPS.keys()}
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            for task, logits in outputs["logits"].items():
                preds = torch.argmax(logits, dim=1).cpu()
                total_correct[task] += (preds == labels[task]).sum().item()
            total_samples += input_ids.size(0)
    
    accuracies = {k: v / total_samples for k, v in total_correct.items()}
    return accuracies

# ---------- Main ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(RAW_DATA)
    print(f"Loaded {len(df)} samples.")
    
    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets & Dataloaders
    train_dataset = EduQADataset(train_df, tokenizer)
    val_dataset = EduQADataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    
    # Model
    model = BanglaEduMultiTaskModel(MODEL_NAME, NUM_CLASSES)
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
    
    # Training loop
    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_acc = evaluate(model, val_loader, device)
        print("Validation Accuracies:")
        for task, acc in val_acc.items():
            print(f"  {task}: {acc:.4f}")
        
        avg_acc = np.mean(list(val_acc.values()))
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"✅ Best model saved with avg acc: {best_acc:.4f}")
    
    # Save tokenizer too
    tokenizer.save_pretrained(MODEL_DIR)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
