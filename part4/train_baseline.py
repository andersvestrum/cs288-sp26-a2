#!/usr/bin/env python3
"""
Part 4 Baseline Training Script

This script demonstrates how to:
1. Train a BPE tokenizer on TinyStories
2. Pretrain a Transformer LM for next-token prediction
3. Fine-tune the model for multiple-choice QA
4. Evaluate using both prompting and fine-tuning approaches

Students can use this as a reference for their implementations.

Usage:
    # First, download datasets
    python part4/setup_datasets.py
    
    # Then run training (use --quick for testing)
    python part4/train_baseline.py --quick      # Quick test (~2 min)
    python part4/train_baseline.py              # Full training (~30 min on GPU)
"""

import argparse
import json
import sys
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from part4.datasets import create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn


# =============================================================================
# Configuration
# =============================================================================

CONFIGS = {
    "quick": {
        # Small config for quick testing
        "pretrain_data": Path(__file__).parent.parent / "part1/fixtures/tinystories_sample_5M.txt",
        "qa_train": Path(__file__).parent / "fixtures/qa_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/qa_dev.json",
        "qa_test": Path(__file__).parent / "fixtures/qa_dev.json",  # no separate test for quick
        "vocab_size": 512,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "context_length": 256,
        "pretrain_epochs": 3,
        "finetune_epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
    },
    "small": {
        # Small model, larger data - ~10M parameters
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "qa_test": Path(__file__).parent / "fixtures/squad_test.json",
        "vocab_size": 4096,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 1024,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 10,
        "batch_size": 32,
        "lr": 3e-4,
    },
    "medium": {
        # Medium model for good quality - ~50M parameters
        "pretrain_data": Path(__file__).parent / "fixtures/tinystories_100k.txt",
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "qa_test": Path(__file__).parent / "fixtures/squad_test.json",
        "vocab_size": 8192,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "context_length": 512,
        "pretrain_epochs": 5,
        "finetune_epochs": 15,
        "batch_size": 16,
        "lr": 1e-4,
    }
}


# =============================================================================
# Step 1: Train BPE Tokenizer
# =============================================================================

def train_tokenizer(pretrain_data: Path, vocab_size: int) -> tuple:
    """
    Train a BPE tokenizer on the pretraining corpus.
    
    Args:
        pretrain_data: Path to training text file
        vocab_size: Target vocabulary size
    
    Returns:
        (tokenizer, vocab, merges)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Training BPE Tokenizer")
    print("=" * 60)
    
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    
    print(f"Input: {pretrain_data}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Train BPE
    vocab, merges = train_bpe(
        input_path=pretrain_data,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    # Create tokenizer
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    
    # Test
    test_text = "Once upon a time, there was a little girl."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTokenizer trained!")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Merges: {len(merges)}")
    print(f"\nTest encoding:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {len(tokens)} tokens")
    print(f"  Decoded: '{decoded}'")
    
    return tokenizer, vocab, merges


# =============================================================================
# Step 2: Pretrain Language Model
# =============================================================================

def pretrain_lm(
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> TransformerLM:
    """
    Pretrain a Transformer language model on TinyStories.
    
    The model learns to predict the next token given previous tokens.
    This gives it general language understanding before fine-tuning.
    
    Args:
        tokenizer: Trained BPE tokenizer
        config: Model and training configuration
        device: Device to train on
    
    Returns:
        Pretrained TransformerLM
    """
    print("\n" + "=" * 60)
    print("STEP 2: Pretraining Language Model")
    print("=" * 60)
    
    # Create model
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  context_length: {config['context_length']}")
    print(f"  Parameters: {num_params:,}")
    
    # Create dataloader
    dataloader = create_pretraining_dataloader(
        file_path=config["pretrain_data"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
        shuffle=True,
    )
    
    print(f"\nTraining data:")
    print(f"  File: {config['pretrain_data']}")
    print(f"  Documents: {len(dataloader.dataset)}")
    print(f"  Batches/epoch: {len(dataloader)}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(dataloader) // 5),
    )
    
    # Train
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
    )
    
    print(f"\nTraining for {config['pretrain_epochs']} epoch(s)...")
    results = trainer.train()
    
    # Test generation
    print("\nGeneration test:")
    for prompt in ["Once upon a time", "The little dog"]:
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=30,
            method="greedy"
        )
        print(f"  '{prompt}' -> '{generated[:80]}...'")
    
    return model


# =============================================================================
# Step 3: Evaluate Zero-Shot Prompting
# =============================================================================

def evaluate_prompting(
    model: TransformerLM,
    tokenizer,
    qa_dev_path: Path,
    device: str = "cpu",
    qa_train_path: Path = None,
) -> dict:
    """
    Evaluate the model using multiple prompting strategies and return the best.
    
    Tries:
      - token scoring with several templates
      - choice_ll scoring with several templates
      - few-shot prompting with choice_ll
    
    Args:
        model: TransformerLM (fine-tuned backbone)
        tokenizer: Tokenizer
        qa_dev_path: Path to validation QA data
        device: Device
        qa_train_path: Optional path to training data for few-shot exemplars
    
    Returns:
        Best evaluation results dict
    """
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Prompting (multiple strategies)")
    print("=" * 60)
    
    # Load data
    with open(qa_dev_path) as f:
        dev_data = json.load(f)
    
    train_data = []
    if qa_train_path and Path(qa_train_path).exists():
        with open(qa_train_path) as f:
            train_data = json.load(f)
    
    print(f"\nValidation examples: {len(dev_data)}")
    
    from part4.prompting import evaluate_prompting as eval_prompt
    
    best_acc = 0.0
    best_results = None
    best_label = ""
    
    # Strategy 1: token scoring with different templates
    for tmpl_name in ["basic", "simple", "instruction", "direct"]:
        template = PromptTemplate(template_name=tmpl_name)
        pipeline = PromptingPipeline(
            model=model, tokenizer=tokenizer,
            template=template, device=device,
            scoring_method="token",
        )
        results = eval_prompt(pipeline, dev_data)
        acc = results["accuracy"]
        label = f"token + {tmpl_name}"
        print(f"  {label}: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_label = label
    
    # Strategy 2: choice_ll scoring with different templates
    for tmpl_name in ["basic", "simple", "instruction", "direct"]:
        template = PromptTemplate(template_name=tmpl_name)
        pipeline = PromptingPipeline(
            model=model, tokenizer=tokenizer,
            template=template, device=device,
            scoring_method="choice_ll",
        )
        results = eval_prompt(pipeline, dev_data)
        acc = results["accuracy"]
        label = f"choice_ll + {tmpl_name}"
        print(f"  {label}: {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_label = label
    
    # Strategy 3: few-shot with choice_ll
    if train_data:
        for num_shots in [1, 3, 5]:
            for tmpl_name in ["basic", "simple"]:
                template = PromptTemplate(
                    template_name=tmpl_name,
                    few_shot_data=train_data[:20],
                    num_shots=num_shots,
                )
                pipeline = PromptingPipeline(
                    model=model, tokenizer=tokenizer,
                    template=template, device=device,
                    scoring_method="choice_ll",
                )
                results = eval_prompt(pipeline, dev_data)
                acc = results["accuracy"]
                label = f"choice_ll + {tmpl_name} + {num_shots}-shot"
                print(f"  {label}: {acc:.2%}")
                if acc > best_acc:
                    best_acc = acc
                    best_results = results
                    best_label = label
    
    print(f"\n  BEST: {best_label} -> {best_acc:.2%}")
    print(f"  Random baseline: 25.00%")
    
    best_results["strategy"] = best_label
    return best_results


# =============================================================================
# Step 4: Fine-tune for QA
# =============================================================================

def finetune_qa(
    pretrained_model: TransformerLM,
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> TransformerForMultipleChoice:
    """
    Fine-tune the pretrained model for multiple-choice QA.
    
    We add a classification head and train the entire model to select
    the correct answer from 4 choices.
    
    Args:
        pretrained_model: Pretrained TransformerLM
        tokenizer: Tokenizer
        config: Training configuration
        device: Device
    
    Returns:
        Fine-tuned QA model
    """
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning for Multiple-Choice QA")
    print("=" * 60)
    
    # Create QA model (wraps the LM with a classification head)
    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",  # Use last token representation
        freeze_backbone=False,  # Fine-tune entire model
    ).to(device)
    
    print(f"\nQA model parameters: {sum(p.numel() for p in qa_model.parameters()):,}")
    
    # Load training data
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
    
    print(f"\nTraining data: {config['qa_train']}")
    print(f"Training examples: {len(train_data)}")
    print(f"Batches/epoch: {len(train_dataloader)}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 2,  # Lower LR for fine-tuning
        weight_decay=0.01,
        warmup_steps=min(50, len(train_dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(train_dataloader) // 5),
    )
    
    # Train
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_dataloader,
        compute_loss_fn=create_qa_loss_fn(device),
    )
    
    print(f"\nFine-tuning for {config['finetune_epochs']} epoch(s)...")
    results = trainer.train()
    
    return qa_model


# =============================================================================
# Step 5: Evaluate Fine-tuned Model
# =============================================================================

def evaluate_finetuned(
    qa_model: TransformerForMultipleChoice,
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the fine-tuned QA model.
    
    Args:
        qa_model: Fine-tuned QA model
        tokenizer: Tokenizer
        config: Configuration
        device: Device
    
    Returns:
        Evaluation results
    """
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Fine-tuned Model")
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
    
    print(f"\nValidation examples: {len(dev_data)}")
    
    # Evaluate
    results = evaluate_qa_model(qa_model, dev_dataloader, device)
    
    print(f"\nFine-tuned model accuracy: {results['accuracy']:.2%}")
    print(f"Random baseline: 25.00%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Part 4 Baseline Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python part4/train_baseline.py --quick     # Quick test (~2 min)
    python part4/train_baseline.py --small     # Small model (~10 min)
    python part4/train_baseline.py --medium    # Medium model (~30 min)
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with tiny model")
    parser.add_argument("--small", action="store_true", help="Small model")
    parser.add_argument("--medium", action="store_true", help="Medium model (default)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    args = parser.parse_args()
    
    # Select config
    if args.quick:
        config_name = "quick"
    elif args.small:
        config_name = "small"
    else:
        config_name = "medium"
    
    config = CONFIGS[config_name]
    
    # Check datasets exist
    if not config["pretrain_data"].exists():
        print(f"Dataset not found: {config['pretrain_data']}")
        if config_name != "quick":
            print("Run: python part4/setup_datasets.py")
            print("Or use: python part4/train_baseline.py --quick")
        return
    
    if not config["qa_train"].exists():
        print(f"Dataset not found: {config['qa_train']}")
        if config_name != "quick":
            print("Run: python part4/setup_datasets.py")
            print("Or use: python part4/train_baseline.py --quick")
        return
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("CS288 Part 4 - Baseline Training")
    print("=" * 60)
    print(f"\nConfiguration: {config_name}")
    print(f"Device: {device}")
    
    # Step 1: Train tokenizer
    # Use bpe_data if specified (faster for large configs), otherwise use pretrain_data
    bpe_data = config.get("bpe_data", config["pretrain_data"])
    tokenizer, vocab, merges = train_tokenizer(
        bpe_data,
        config["vocab_size"]
    )
    
    # Step 2: Pretrain LM
    pretrained_model = pretrain_lm(tokenizer, config, device)
    
    # Step 3: Fine-tune for QA
    qa_model = finetune_qa(pretrained_model, tokenizer, config, device)
    
    # Step 4: Evaluate prompting on fine-tuned model (dev set — find best strategy)
    # Use the fine-tuned backbone (qa_model.transformer) for prompting
    prompting_dev_results = evaluate_prompting(
        qa_model.transformer, tokenizer,
        config["qa_dev"], device,
        qa_train_path=config["qa_train"],
    )
    
    # Step 5: Evaluate fine-tuned model on dev set (classification head)
    finetuned_dev_results = evaluate_finetuned(qa_model, tokenizer, config, device)
    
    # =========================================================================
    # Step 6: Generate TEST SET predictions for Gradescope submission
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Generating TEST SET predictions for submission")
    print("=" * 60)
    
    qa_test_path = config.get("qa_test", config["qa_dev"])
    with open(qa_test_path) as f:
        test_data = json.load(f)
    print(f"\nTest examples: {len(test_data)}")
    
    # --- Fine-tuned test predictions ---
    print("\nGenerating fine-tuned test predictions...")
    test_dataloader = create_qa_dataloader(
        data=test_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False,
    )
    finetuned_test_results = evaluate_qa_model(qa_model, test_dataloader, device)
    print(f"  Fine-tuned test accuracy: {finetuned_test_results['accuracy']:.2%}")
    
    # --- Prompting test predictions using the best strategy found on dev ---
    best_strategy = prompting_dev_results.get("strategy", "choice_ll + basic")
    print(f"\nGenerating prompting test predictions using best dev strategy: {best_strategy}")
    
    # Parse best strategy to reconstruct the pipeline
    from part4.prompting import evaluate_prompting as eval_prompt
    
    train_data = []
    if config["qa_train"].exists():
        with open(config["qa_train"]) as f:
            train_data = json.load(f)
    
    # Parse strategy string: "choice_ll + basic + 3-shot" or "token + simple"
    parts = [p.strip() for p in best_strategy.split("+")]
    scoring_method = parts[0] if parts else "choice_ll"
    tmpl_name = parts[1] if len(parts) > 1 else "basic"
    num_shots = 0
    if len(parts) > 2 and "-shot" in parts[2]:
        num_shots = int(parts[2].replace("-shot", ""))
    
    template_kwargs = {"template_name": tmpl_name}
    if num_shots > 0 and train_data:
        template_kwargs["few_shot_data"] = train_data[:20]
        template_kwargs["num_shots"] = num_shots
    
    template = PromptTemplate(**template_kwargs)
    pipeline = PromptingPipeline(
        model=qa_model.transformer, tokenizer=tokenizer,
        template=template, device=device,
        scoring_method=scoring_method,
    )
    prompting_test_results = eval_prompt(pipeline, test_data)
    print(f"  Prompting test accuracy: {prompting_test_results['accuracy']:.2%}")
    
    # Summary (dev set results for reference)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nConfiguration: {config_name}")
    print(f"Model parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")
    print(f"\nDEV SET results (used for strategy selection):")
    print(f"  Prompting approach:    {prompting_dev_results['accuracy']:.2%}")
    print(f"  Classification head:   {finetuned_dev_results['accuracy']:.2%}")
    print(f"\nTEST SET results (submitted to Gradescope):")
    print(f"  Prompting approach:    {prompting_test_results['accuracy']:.2%}")
    print(f"  Classification head:   {finetuned_test_results['accuracy']:.2%}")
    print(f"  Random baseline:       25.00%")
    
    # Calculate improvement on TEST set
    prompting_boost = prompting_test_results['accuracy'] - finetuned_test_results['accuracy']
    print(f"\n  Prompting boost over fine-tuned (test): {prompting_boost:+.2%}")
    if prompting_boost >= 0.04:
        print(f"  (4%+ boost = full prompting score)")
    elif prompting_boost > 0:
        print(f"  (Need 4% boost for full prompting score)")
    else:
        print(f"  (Prompting should beat fine-tuned model)")
    
    # Save TEST predictions to JSON files for Gradescope grading
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save fine-tuned TEST predictions
    finetuned_output = {
        "predictions": finetuned_test_results.get("predictions", []),
        "accuracy": finetuned_test_results["accuracy"],
        "dev_accuracy": finetuned_dev_results["accuracy"],
        "config": config_name,
        "split": "test",
    }
    finetuned_path = output_dir / "finetuned_predictions.json"
    with open(finetuned_path, "w") as f:
        json.dump(finetuned_output, f, indent=2)
    
    # Save prompting TEST predictions
    prompting_output = {
        "predictions": prompting_test_results.get("predictions", []),
        "accuracy": prompting_test_results["accuracy"],
        "dev_accuracy": prompting_dev_results["accuracy"],
        "config": config_name,
        "strategy": best_strategy,
        "split": "test",
    }
    prompting_path = output_dir / "prompting_predictions.json"
    with open(prompting_path, "w") as f:
        json.dump(prompting_output, f, indent=2)
    
    print(f"\nTEST predictions saved to:")
    print(f"  {finetuned_path}")
    print(f"  {prompting_path}")
    
    # Print grading info (estimated from test results)
    print("\n" + "=" * 60)
    print("ESTIMATED GRADING (based on test set)")
    print("=" * 60)
    finetuned_score = max(0, min(1, (finetuned_test_results['accuracy'] - 0.30) / 0.20))
    prompting_score = max(0, min(1, prompting_boost / 0.04)) if prompting_boost > 0 else 0
    total_score = 0.5 * finetuned_score + 0.5 * prompting_score
    
    print(f"\nFine-tuned score:  {finetuned_score:.0%} (30%=0pts, 50%=full)")
    print(f"Prompting score:   {prompting_score:.0%} (0% boost=0pts, 4% boost=full)")
    print(f"Total Part 4:      {total_score:.0%}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
