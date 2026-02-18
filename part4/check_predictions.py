#!/usr/bin/env python3
"""Check prediction accuracy against squad_dev.json ground truth labels."""
import json
import os


def check_accuracy(pred_path, data_path):
    with open(data_path) as f:
        data = json.load(f)
    with open(pred_path) as f:
        pred_json = json.load(f)

    preds = pred_json["predictions"]
    saved_acc = pred_json.get("accuracy", None)

    labeled = [(i, ex) for i, ex in enumerate(data) if "answer" in ex]
    print(f"  Data entries:        {len(data)}")
    print(f"  Entries with labels: {len(labeled)}")
    print(f"  Predictions count:   {len(preds)}")
    if saved_acc is not None:
        print(f"  Saved accuracy:      {saved_acc:.4f}")

    if len(preds) != len(data):
        print(f"  ⚠️  MISMATCH: {len(preds)} predictions vs {len(data)} data entries!")

    correct = 0
    total = 0
    wrong = []

    for idx, ex in labeled:
        if idx >= len(preds):
            print(f"  ⚠️  Index {idx} out of range for predictions")
            break
        pred = preds[idx]
        label = ex["answer"]
        total += 1
        if pred == label:
            correct += 1
        else:
            wrong.append({
                "idx": idx,
                "id": ex.get("id", "?"),
                "question": ex.get("question", "?")[:80],
                "pred": pred,
                "label": label,
                "pred_text": ex["choices"][pred] if "choices" in ex and pred < len(ex["choices"]) else "?",
                "label_text": ex["choices"][label] if "choices" in ex and label < len(ex["choices"]) else "?",
            })

    acc = correct / total if total > 0 else 0
    print(f"\n  Computed accuracy:   {correct}/{total} = {acc:.4f} ({acc:.2%})")

    if saved_acc is not None and abs(acc - saved_acc) > 1e-6:
        print(f"  ⚠️  MISMATCH with saved accuracy ({saved_acc:.4f})!")
    elif saved_acc is not None:
        print(f"  ✅ Matches saved accuracy")

    # Show first 5 wrong predictions
    if wrong:
        print(f"\n  First 5 wrong (of {len(wrong)}):")
        for w in wrong[:5]:
            print(f"    [{w['idx']}] {w['question']}")
            print(f"         pred={w['pred']} ({w['pred_text']})")
            print(f"        label={w['label']} ({w['label_text']})")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(base, "fixtures", "squad_dev.json")
    outputs_dir = os.path.join(base, "outputs")

    for name in ["finetuned_predictions.json", "prompting_predictions.json"]:
        pred_path = os.path.join(outputs_dir, name)
        print("=" * 60)
        print(f"Checking: {name}")
        print("=" * 60)
        if not os.path.exists(pred_path):
            print(f"  ⚠️  Not found: {pred_path}\n")
            continue
        check_accuracy(pred_path, dev_path)
        print()
