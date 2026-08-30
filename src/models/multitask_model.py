import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BanglaEduMultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes_dict, dropout=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        
        # Classification heads
        self.subject_head = nn.Linear(self.hidden_size, num_classes_dict["subject"])
        self.topic_head = nn.Linear(self.hidden_size, num_classes_dict["topic"])
        self.qtype_head = nn.Linear(self.hidden_size, num_classes_dict["question_type"])
        self.diff_head = nn.Linear(self.hidden_size, num_classes_dict["difficulty"])
        self.cog_head = nn.Linear(self.hidden_size, num_classes_dict["cognitive_level"])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        pooled = self.dropout(pooled)
        
        logits_subj = self.subject_head(pooled)
        logits_topic = self.topic_head(pooled)
        logits_qtype = self.qtype_head(pooled)
        logits_diff = self.diff_head(pooled)
        logits_cog = self.cog_head(pooled)
        
        logits_dict = {
            "subject": logits_subj,
            "topic": logits_topic,
            "question_type": logits_qtype,
            "difficulty": logits_diff,
            "cognitive_level": logits_cog,
        }
        
        # If labels are provided, compute multi-task loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0.0
            for task, logits in logits_dict.items():
                if task in labels:
                    total_loss += loss_fct(logits, labels[task])
            return {"loss": total_loss, "logits": logits_dict}
        
        return {"logits": logits_dict}
