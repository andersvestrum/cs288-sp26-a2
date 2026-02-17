"""
Multiple-choice QA model.
Example submission.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part2.model import Linear


class TransformerForMultipleChoice(nn.Module):
    def __init__(self, transformer_lm, hidden_size: int, num_choices: int = 4, pooling: str = "last", freeze_backbone: bool = False):
        super().__init__()
        self.transformer = transformer_lm
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        self.pooling = pooling
        self.classifier = Linear(hidden_size, 1)
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
    
    def _get_hidden_states(self, input_ids: Tensor) -> Tensor:
        x = self.transformer.token_embeddings(input_ids)
        for layer in self.transformer.layers:
            x = layer(x)
        x = self.transformer.final_ln(x)
        return x
    
    def _pool(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if self.pooling == "last":
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths]
            return hidden_states[:, -1]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            return hidden_states.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                hidden_states = hidden_states.masked_fill(~mask.bool(), float('-inf'))
            return hidden_states.max(dim=1).values
        raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, num_choices, seq_len = input_ids.shape
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len) if attention_mask is not None else None
        hidden_states = self._get_hidden_states(input_ids_flat)
        pooled = self._pool(hidden_states, attention_mask_flat)
        logits = self.classifier(pooled).squeeze(-1)
        return logits.view(batch_size, num_choices)
    
    @torch.no_grad()
    def predict(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        self.eval()
        return self.forward(input_ids, attention_mask).argmax(dim=-1)


def evaluate_qa_model(model, dataloader, device: str = "cuda") -> dict:
    model.eval()
    model.to(device)
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            predictions = model.predict(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if l >= 0 and p == l)
    total = sum(1 for l in all_labels if l >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": all_predictions, "labels": all_labels}
