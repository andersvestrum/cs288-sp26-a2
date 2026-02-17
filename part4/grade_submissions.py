#!/usr/bin/env python3
"""
Grading script for Part 4 submissions.

Evaluates student predictions against the validation set and computes scores.

Grading Rubric:
---------------
1. Fine-tuned Model (50% of part 4 grade):
   - 30% accuracy = 0 points
   - 50% accuracy = full points
   - Linear interpolation between 30% and 50%
   - Below 30% = 0 points
   - Above 50% = full points (capped)

2. Prompting Model (50% of part 4 grade):
   - Must achieve at least 4% improvement over fine-tuned model for full points
   - 0% improvement = 0 points
   - 4%+ improvement = full points
   - Linear interpolation between 0% and 4%
   - Negative improvement = 0 points

Expected JSON format:
    {
        "predictions": [0, 1, 2, 3, ...],  # List of predicted answer indices
        "accuracy": 0.45  # Optional: self-reported accuracy
    }

Usage:
    python grade_submissions.py --finetuned student_finetuned.json --prompting student_prompting.json --validation val_data.json
    python grade_submissions.py --submissions_dir ./submissions/ --validation val_data.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """Calculate accuracy between predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels")
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels) if labels else 0.0


def score_finetuned(accuracy: float, min_acc: float = 0.30, max_acc: float = 0.50) -> float:
    """
    Calculate score for fine-tuned model.
    
    Args:
        accuracy: Model accuracy (0.0 to 1.0)
        min_acc: Minimum accuracy for 0 points (default: 30%)
        max_acc: Accuracy for full points (default: 50%)
    
    Returns:
        Score from 0.0 to 1.0 (1.0 = full points)
    """
    if accuracy <= min_acc:
        return 0.0
    elif accuracy >= max_acc:
        return 1.0
    else:
        # Linear interpolation
        return (accuracy - min_acc) / (max_acc - min_acc)


def score_prompting(prompting_acc: float, finetuned_acc: float, required_boost: float = 0.04) -> float:
    """
    Calculate score for prompting model.
    
    Args:
        prompting_acc: Prompting model accuracy (0.0 to 1.0)
        finetuned_acc: Fine-tuned model accuracy (0.0 to 1.0)
        required_boost: Required improvement for full points (default: 4%)
    
    Returns:
        Score from 0.0 to 1.0 (1.0 = full points)
    """
    boost = prompting_acc - finetuned_acc
    
    if boost <= 0:
        return 0.0
    elif boost >= required_boost:
        return 1.0
    else:
        # Linear interpolation
        return boost / required_boost


def grade_submission(
    finetuned_predictions: List[int],
    prompting_predictions: List[int],
    labels: List[int],
    finetuned_weight: float = 0.5,
    prompting_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Grade a student submission.
    
    Args:
        finetuned_predictions: Predictions from fine-tuned model
        prompting_predictions: Predictions from prompting model
        labels: Ground truth labels
        finetuned_weight: Weight for fine-tuned score (default: 50%)
        prompting_weight: Weight for prompting score (default: 50%)
    
    Returns:
        Dictionary with detailed grading results
    """
    # Calculate accuracies
    finetuned_acc = calculate_accuracy(finetuned_predictions, labels)
    prompting_acc = calculate_accuracy(prompting_predictions, labels)
    
    # Calculate component scores
    finetuned_score = score_finetuned(finetuned_acc)
    prompting_score = score_prompting(prompting_acc, finetuned_acc)
    
    # Calculate weighted total
    total_score = (finetuned_weight * finetuned_score + 
                   prompting_weight * prompting_score)
    
    return {
        "finetuned": {
            "accuracy": finetuned_acc,
            "accuracy_pct": f"{finetuned_acc * 100:.2f}%",
            "score": finetuned_score,
            "score_pct": f"{finetuned_score * 100:.2f}%",
            "weight": finetuned_weight,
            "weighted_score": finetuned_weight * finetuned_score,
        },
        "prompting": {
            "accuracy": prompting_acc,
            "accuracy_pct": f"{prompting_acc * 100:.2f}%",
            "boost_over_finetuned": prompting_acc - finetuned_acc,
            "boost_pct": f"{(prompting_acc - finetuned_acc) * 100:.2f}%",
            "score": prompting_score,
            "score_pct": f"{prompting_score * 100:.2f}%",
            "weight": prompting_weight,
            "weighted_score": prompting_weight * prompting_score,
        },
        "total_score": total_score,
        "total_score_pct": f"{total_score * 100:.2f}%",
        "num_examples": len(labels),
    }


def print_grade_report(results: Dict[str, Any], student_name: str = "Student"):
    """Print a formatted grade report."""
    print("=" * 60)
    print(f"GRADE REPORT: {student_name}")
    print("=" * 60)
    
    ft = results["finetuned"]
    pr = results["prompting"]
    
    print(f"\n1. Fine-tuned Model ({ft['weight']*100:.0f}% weight)")
    print(f"   Accuracy: {ft['accuracy_pct']} ({ft['accuracy']*100:.2f}%)")
    print(f"   Score: {ft['score_pct']} (30% = 0pts, 50% = full pts)")
    print(f"   Weighted: {ft['weighted_score']*100:.2f}%")
    
    print(f"\n2. Prompting Model ({pr['weight']*100:.0f}% weight)")
    print(f"   Accuracy: {pr['accuracy_pct']}")
    print(f"   Boost over fine-tuned: {pr['boost_pct']} (need 4% for full pts)")
    print(f"   Score: {pr['score_pct']}")
    print(f"   Weighted: {pr['weighted_score']*100:.2f}%")
    
    print(f"\n" + "=" * 60)
    print(f"TOTAL SCORE: {results['total_score_pct']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Grade Part 4 submissions")
    parser.add_argument("--finetuned", type=str, help="Path to fine-tuned predictions JSON")
    parser.add_argument("--prompting", type=str, help="Path to prompting predictions JSON")
    parser.add_argument("--validation", type=str, required=True, help="Path to validation data JSON")
    parser.add_argument("--submissions_dir", type=str, help="Directory containing submissions")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Load validation data
    val_data = load_json(args.validation)
    
    # Extract labels from validation data
    if isinstance(val_data, list):
        labels = [ex.get("answer", ex.get("label", -1)) for ex in val_data]
    elif "labels" in val_data:
        labels = val_data["labels"]
    elif "answers" in val_data:
        labels = val_data["answers"]
    else:
        raise ValueError("Cannot find labels in validation data")
    
    # Filter out invalid labels
    valid_indices = [i for i, l in enumerate(labels) if l >= 0]
    labels = [labels[i] for i in valid_indices]
    
    if args.submissions_dir:
        # Grade all submissions in directory
        submissions_dir = Path(args.submissions_dir)
        results = {}
        
        for student_dir in submissions_dir.iterdir():
            if not student_dir.is_dir():
                continue
            
            finetuned_path = student_dir / "finetuned_predictions.json"
            prompting_path = student_dir / "prompting_predictions.json"
            
            if not finetuned_path.exists() or not prompting_path.exists():
                print(f"Skipping {student_dir.name}: missing prediction files")
                continue
            
            finetuned_data = load_json(finetuned_path)
            prompting_data = load_json(prompting_path)
            
            ft_preds = finetuned_data.get("predictions", finetuned_data)
            pr_preds = prompting_data.get("predictions", prompting_data)
            
            # Filter predictions to match valid labels
            ft_preds = [ft_preds[i] for i in valid_indices if i < len(ft_preds)]
            pr_preds = [pr_preds[i] for i in valid_indices if i < len(pr_preds)]
            
            result = grade_submission(ft_preds, pr_preds, labels)
            results[student_dir.name] = result
            print_grade_report(result, student_dir.name)
            print()
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    else:
        # Grade single submission
        if not args.finetuned or not args.prompting:
            parser.error("Must provide --finetuned and --prompting, or --submissions_dir")
        
        finetuned_data = load_json(args.finetuned)
        prompting_data = load_json(args.prompting)
        
        ft_preds = finetuned_data.get("predictions", finetuned_data)
        pr_preds = prompting_data.get("predictions", prompting_data)
        
        # Filter predictions to match valid labels
        ft_preds = [ft_preds[i] for i in valid_indices if i < len(ft_preds)]
        pr_preds = [pr_preds[i] for i in valid_indices if i < len(pr_preds)]
        
        result = grade_submission(ft_preds, pr_preds, labels)
        print_grade_report(result)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
