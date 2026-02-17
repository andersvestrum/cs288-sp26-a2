#!/usr/bin/env python3
"""
Part 4 Model Evaluation Script

This script:
1. Trains a BPE tokenizer on TinyStories
2. Pretrains a Transformer LM on TinyStories
3. Fine-tunes for multiple-choice QA on SQuAD
4. Evaluates both prompting and fine-tuned models on validation set

Datasets (run setup_datasets.py first):
- Pretraining: TinyStories (~2.1M stories, ~500MB)
- QA: SQuAD v1.1 converted to multiple choice (10k train, 2k val)

Usage:
    python part4/setup_datasets.py   # Download datasets first
    python part4/evaluate_models.py  # Run evaluation

Options:
    --quick    Use smaller datasets for quick testing
    --full     Use full TinyStories + SQuAD (default)
"""

import sys
import json
import argparse
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from part4.datasets import PretrainingDataset, MultipleChoiceQADataset, create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import greedy_decode, generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting
from part4.trainer import Trainer, TrainingConfig

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PART1_FIXTURES = Path(__file__).parent.parent / "part1" / "fixtures"

# Full datasets (run setup_datasets.py to download)
PRETRAIN_DATA_FULL = FIXTURES_DIR / "tinystories_full.txt"
PRETRAIN_DATA_100K = FIXTURES_DIR / "tinystories_100k.txt"
SQUAD_TRAIN = FIXTURES_DIR / "squad_train.json"
SQUAD_DEV = FIXTURES_DIR / "squad_dev.json"
SQUAD_TEST = FIXTURES_DIR / "squad_test.json"

# Small datasets (included in repo)
PRETRAIN_DATA_SMALL = PART1_FIXTURES / "tinystories_sample_5M.txt"
QA_TRAIN_SMALL = FIXTURES_DIR / "qa_train.json"
QA_DEV_SMALL = FIXTURES_DIR / "qa_dev.json"


def get_config(mode: str = "full"):
    """Get configuration based on mode."""
    if mode == "quick":
        return {
            "pretrain_data": PRETRAIN_DATA_SMALL,
            "qa_train": QA_TRAIN_SMALL,
            "qa_dev": QA_DEV_SMALL,
            "vocab_size": 512,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 2,
            "d_ff": 128,
            "context_length": 128,
            "pretrain_epochs": 3,
            "finetune_epochs": 5,
            "batch_size": 4,
            "learning_rate": 1e-3,
        }
    else:  # full mode
        return {
            "pretrain_data": PRETRAIN_DATA_100K if PRETRAIN_DATA_100K.exists() else PRETRAIN_DATA_FULL,
            "qa_train": SQUAD_TRAIN if SQUAD_TRAIN.exists() else QA_TRAIN_SMALL,
            "qa_dev": SQUAD_DEV if SQUAD_DEV.exists() else QA_DEV_SMALL,
            "vocab_size": 4096,
            "d_model": 256,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 1024,
            "context_length": 256,
            "pretrain_epochs": 1,  # 1 epoch over 100k stories is substantial
            "finetune_epochs": 3,
            "batch_size": 16,
            "learning_rate": 3e-4,
        }


def train_tokenizer(config: dict) -> tuple:
    """Train BPE tokenizer on TinyStories."""
    print("=" * 60)
    print("Step 1: Training BPE Tokenizer")
    print("=" * 60)
    
    pretrain_data = config["pretrain_data"]
    vocab_size = config["vocab_size"]
    
    print(f"Training data: {pretrain_data}")
    print(f"Target vocab size: {vocab_size}")
    
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab, merges = train_bpe(
        input_path=pretrain_data,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    
    # Test tokenizer
    test_text = "Once upon a time, there was a little girl."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Test: '{test_text}'")
    print(f"  -> {len(tokens)} tokens")
    print(f"  -> decoded: '{decoded}'")
    print()
    
    return tokenizer, vocab, merges


def pretrain_model(tokenizer, config: dict, device: str = "cpu") -> TransformerLM:
    """Pretrain Transformer LM on TinyStories."""
    print("=" * 60)
    print("Step 2: Pretraining Transformer LM")
    print("=" * 60)
    
    # Model config
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)
    
    # Create dataloader
    dataloader = create_pretraining_dataloader(
        file_path=config["pretrain_data"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
        shuffle=True,
    )
    
    print(f"Model config: d_model={config['d_model']}, layers={config['num_layers']}, heads={config['num_heads']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(dataloader.dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 10),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(dataloader) // 10),
    )
    
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
    )
    
    # Train
    print("\nTraining...")
    results = trainer.train()
    
    for epoch, loss in enumerate(results["train_losses"]):
        print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")
    
    # Test generation
    print("\nGeneration test:")
    prompt = "Once upon a time"
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=30, method="greedy")
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{generated}'")
    print()
    
    return model


def evaluate_prompting_approach(model, tokenizer, config: dict, device: str = "cpu") -> dict:
    """Evaluate the prompting approach on QA."""
    print("=" * 60)
    print("Step 3: Evaluating Prompting Approach (Zero-Shot)")
    print("=" * 60)
    
    # Load QA data
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    
    print(f"Validation examples: {len(dev_data)}")
    
    # Create prompting pipeline
    template = PromptTemplate(template_name="simple")
    pipeline = PromptingPipeline(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=device,
    )
    
    # Evaluate
    results = evaluate_prompting(pipeline, dev_data)
    
    print(f"Prompting accuracy: {results['accuracy']:.2%}")
    print(f"(Random baseline: 25.00%)")
    print()
    
    return results


def finetune_qa_model(pretrained_model, tokenizer, config: dict, device: str = "cpu") -> TransformerForMultipleChoice:
    """Fine-tune for multiple-choice QA on SQuAD."""
    print("=" * 60)
    print("Step 4: Fine-tuning QA Model on SQuAD")
    print("=" * 60)
    
    # Create QA model
    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",
        freeze_backbone=False,  # Fine-tune entire model
    ).to(device)
    
    print(f"QA model parameters: {sum(p.numel() for p in qa_model.parameters()):,}")
    
    # Load data
    with open(config["qa_train"]) as f:
        train_data = json.load(f)
    
    train_dataloader = create_qa_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=True,
    )
    
    print(f"Training data: {config['qa_train']}")
    print(f"Training examples: {len(train_data)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Training
    from part4.trainer import create_qa_loss_fn
    
    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_steps=min(50, len(train_dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(train_dataloader) // 10),
    )
    
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_dataloader,
        compute_loss_fn=create_qa_loss_fn(device),
    )
    
    print("\nTraining...")
    results = trainer.train()
    
    for epoch, loss in enumerate(results["train_losses"]):
        print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")
    print()
    
    return qa_model


def evaluate_finetuned_model(qa_model, tokenizer, config: dict, device: str = "cpu") -> dict:
    """Evaluate fine-tuned QA model on SQuAD validation."""
    print("=" * 60)
    print("Step 5: Evaluating Fine-tuned Model on SQuAD Dev")
    print("=" * 60)
    
    # Load validation data
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    
    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False,
    )
    
    print(f"Validation data: {config['qa_dev']}")
    print(f"Validation examples: {len(dev_data)}")
    
    results = evaluate_qa_model(qa_model, dev_dataloader, device)
    
    print(f"Fine-tuned model accuracy: {results['accuracy']:.2%}")
    print(f"(Random baseline: 25.00%)")
    print()
    
    return results


def run_tests():
    """Run the part4 tests."""
    print("=" * 60)
    print("Step 6: Running Tests")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "part4/tests/test_part4.py", "-v"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="CS288 Part 4 Evaluation")
    parser.add_argument("--quick", action="store_true", help="Use smaller datasets for quick testing")
    parser.add_argument("--full", action="store_true", help="Use full TinyStories + SQuAD (default)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest")
    args = parser.parse_args()
    
    mode = "quick" if args.quick else "full"
    config = get_config(mode)
    
    print("\n" + "=" * 60)
    print("CS288 Assignment 2 - Part 4 Evaluation")
    print("=" * 60)
    print(f"\nMode: {mode.upper()}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Check if datasets exist
    if not config["pretrain_data"].exists():
        print(f"\nDataset not found: {config['pretrain_data']}")
        print("Run 'python part4/setup_datasets.py' to download datasets.")
        print("Or use --quick mode for small bundled datasets.")
        return
    
    if not config["qa_train"].exists():
        print(f"\nDataset not found: {config['qa_train']}")
        print("Run 'python part4/setup_datasets.py' to download datasets.")
        print("Or use --quick mode for small bundled datasets.")
        return
    
    print(f"\nPretraining data: {config['pretrain_data']}")
    print(f"QA training data: {config['qa_train']}")
    print()
    
    # Step 1: Train tokenizer
    tokenizer, vocab, merges = train_tokenizer(config)
    
    # Step 2: Pretrain LM
    pretrained_model = pretrain_model(tokenizer, config, device)
    
    # Step 3: Evaluate prompting (zero-shot)
    prompting_results = evaluate_prompting_approach(pretrained_model, tokenizer, config, device)
    
    # Step 4: Fine-tune for QA
    qa_model = finetune_qa_model(pretrained_model, tokenizer, config, device)
    
    # Step 5: Evaluate fine-tuned model
    finetuned_results = evaluate_finetuned_model(qa_model, tokenizer, config, device)
    
    # Step 6: Run tests (optional)
    if not args.skip_tests:
        tests_passed = run_tests()
    else:
        tests_passed = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mode:                 {mode}")
    print(f"Model size:           {sum(p.numel() for p in pretrained_model.parameters()):,} params")
    print(f"Prompting accuracy:   {prompting_results['accuracy']:.2%}")
    print(f"Fine-tuned accuracy:  {finetuned_results['accuracy']:.2%}")
    if tests_passed is not None:
        print(f"Tests passed:         {'Yes' if tests_passed else 'No'}")
    print()
    
    improvement = finetuned_results['accuracy'] - prompting_results['accuracy']
    if improvement > 0:
        print(f"Fine-tuning improved by {improvement:.1%} over zero-shot prompting!")
    else:
        print("Note: Fine-tuning may need more epochs or larger model to outperform prompting.")


if __name__ == "__main__":
    main()
