"""
Training utilities for pre-training and fine-tuning.

Provides:
  - TrainingConfig: Dataclass for hyperparameters
  - Trainer: Training loop with AdamW + cosine schedule + gradient clipping
             + optional gradient accumulation
  - create_qa_loss_fn: Loss function factory for QA fine-tuning
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import time
import sys

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    patience: Optional[int] = None


class Trainer:
    """
    Generic trainer for both LM pre-training and QA fine-tuning.

    Uses AdamW with linear warmup followed by cosine decay.
    Supports a custom loss function for different tasks and
    gradient accumulation for memory-constrained fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        compute_loss_fn: Optional[Callable] = None,
        grad_accum_steps: int = 1,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        self.grad_accum_steps = max(1, grad_accum_steps)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler counts *optimizer* steps (after accumulation)
        steps_per_epoch = len(train_dataloader) // self.grad_accum_steps
        total_optim_steps = steps_per_epoch * config.num_epochs
        if config.warmup_steps > 0 and total_optim_steps > config.warmup_steps:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
            main = CosineAnnealingLR(self.optimizer, T_max=max(1, total_optim_steps - config.warmup_steps))
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[config.warmup_steps])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_optim_steps))

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        """Language-modeling loss (next-token prediction)."""
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

    def train_epoch(self) -> float:
        """Run one training epoch with progress bar, logging, and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        loader = self.train_dataloader
        if tqdm is not None:
            loader = tqdm(loader, desc="Training", leave=True)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            loss = self.compute_loss_fn(batch, self.model)

            # Scale loss by accumulation steps so the total gradient is averaged
            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps

            loss.backward()

            total_loss += loss.item() * self.grad_accum_steps  # track unscaled loss
            num_batches += 1

            # Optimizer step every grad_accum_steps micro-batches (or at end of epoch)
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            if tqdm is not None and hasattr(loader, 'set_postfix'):
                loader.set_postfix(
                    loss=f"{loss.item() * self.grad_accum_steps:.4f}",
                    avg_loss=f"{total_loss / num_batches:.4f}",
                    step=self.global_step,
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on the validation set and return average loss."""
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self) -> Dict[str, Any]:
        """Full training loop over all epochs."""
        accum_msg = f", grad_accum={self.grad_accum_steps}" if self.grad_accum_steps > 1 else ""
        print(f"\nStarting training: {self.config.num_epochs} epochs, "
              f"lr={self.config.learning_rate}, device={self.config.device}{accum_msg}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"  Train loss: {train_loss:.4f}")

            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"  Val loss:   {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


# =============================================================================
# QA Loss
# =============================================================================

def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    """Cross-entropy loss for multiple-choice QA classification."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, attention_mask)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    """Create a QA loss function bound to a specific device."""
    return lambda batch, model: compute_qa_loss(batch, model, device)
