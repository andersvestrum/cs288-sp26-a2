#!/usr/bin/env python3
"""
Download and prepare datasets for Part 4.

Datasets:
- TinyStories: ~2.1M short children's stories for pretraining
- SQuAD v1.1: ~100k QA examples for fine-tuning

Usage:
    python part4/setup_datasets.py
"""

import json
import os
import sys
from pathlib import Path

# Remove current directory from path to avoid importing local datasets.py
# instead of the HuggingFace datasets library
_this_dir = str(Path(__file__).parent)
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

# Try to import datasets, install if needed
try:
    import datasets as hf_datasets
    load_dataset = hf_datasets.load_dataset
except ImportError:
    print("Installing 'datasets' library...")
    os.system("pip install datasets")
    import datasets as hf_datasets
    load_dataset = hf_datasets.load_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def download_tinystories():
    """Download TinyStories dataset for pretraining."""
    print("=" * 60)
    print("Downloading TinyStories dataset...")
    print("=" * 60)
    
    # Load TinyStories from HuggingFace
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    print(f"Total stories: {len(dataset):,}")
    
    # Save as text file with <|endoftext|> separator
    output_path = FIXTURES_DIR / "tinystories_full.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            story = example["text"].strip()
            f.write(story)
            f.write("\n<|endoftext|>\n")
            
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1:,} stories...")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Also create a smaller subset for quick testing
    subset_path = FIXTURES_DIR / "tinystories_100k.txt"
    with open(subset_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            if i >= 100000:
                break
            story = example["text"].strip()
            f.write(story)
            f.write("\n<|endoftext|>\n")
    
    print(f"Also created 100k subset: {subset_path}")
    
    return output_path


def download_squad():
    """Download SQuAD v1.1 dataset for QA fine-tuning."""
    print("\n" + "=" * 60)
    print("Checking SQuAD datasets...")
    print("=" * 60)
    
    train_path = FIXTURES_DIR / "squad_train.json"
    val_path = FIXTURES_DIR / "squad_dev.json"
    test_path = FIXTURES_DIR / "squad_test.json"
    
    # Use the committed JSON files as the canonical data source.
    # NEVER regenerate them — predictions must match these exact files.
    if train_path.exists() and val_path.exists() and test_path.exists():
        with open(train_path) as f:
            n_train = len(json.load(f))
        with open(val_path) as f:
            n_val = len(json.load(f))
        with open(test_path) as f:
            n_test = len(json.load(f))
        print(f"\nSQuAD datasets already exist — using committed versions:")
        print(f"  Training:   {train_path} ({n_train:,} examples)")
        print(f"  Validation: {val_path} ({n_val:,} examples)")
        print(f"  Test:       {test_path} ({n_test:,} examples)")
        return train_path, val_path, test_path
    
    # Only reach here if files are missing (shouldn't happen after git clone)
    raise FileNotFoundError(
        "SQuAD fixture files not found! These should be committed in the repo.\n"
        f"Expected: {train_path}, {val_path}, {test_path}"
    )


def main():
    print("\n" + "=" * 60)
    print("CS288 Part 4 - Dataset Setup")
    print("=" * 60 + "\n")
    
    # Download TinyStories
    tinystories_path = download_tinystories()
    
    # Download SQuAD
    squad_paths = download_squad()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nDatasets ready in:", FIXTURES_DIR)
    print("\nRecommended usage:")
    print("  - Pretraining: tinystories_full.txt (full) or tinystories_100k.txt (quick)")
    print("  - Fine-tuning: squad_train.json (10k examples)")
    print("  - Validation:  squad_dev.json (2k examples)")
    print("  - Testing:     squad_test.json (1k examples)")


if __name__ == "__main__":
    main()
