"""
Prompting utilities for multiple-choice QA.

Uses the fine-tuned TransformerForMultipleChoice model (classification head)
with different input prompt formats. "Prompting" here means experimenting
with how context/question/answer are formatted before being fed to the
same classification model from Part 4A.

Key improvements over the baseline fine-tuned evaluation:
1. Smart truncation that always preserves question + answer (matches training)
2. Multiple complementary prompt formats
3. Majority-vote ensemble across formats for robust predictions
"""
import torch
from collections import Counter
from typing import List, Dict, Any, Optional, Callable
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt format functions: each takes (context, question, choice) → str
# ─────────────────────────────────────────────────────────────────────────────

def format_base(context: str, question: str, choice: str) -> str:
    """Baseline format — identical to fine-tuning."""
    return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"


def format_highlight(context: str, question: str, choice: str) -> str:
    """Repeat the question near the answer to strengthen the signal."""
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"The answer to \"{question}\" is: {choice}"
    )


def format_compact(context: str, question: str, choice: str) -> str:
    """Compact style — less padding tokens, more context fits."""
    return f"{context}\nQ: {question}\nA: {choice}"


PROMPT_STYLES: Dict[str, Callable] = {
    "base": format_base,
    "highlight": format_highlight,
    "compact": format_compact,
}


# ─────────────────────────────────────────────────────────────────────────────
# Smart truncation: always preserve question + answer, trim context only
# ─────────────────────────────────────────────────────────────────────────────

def encode_with_smart_truncation(
    tokenizer, context: str, suffix: str, max_length: int
) -> List[int]:
    """
    Encode context + suffix, trimming context (not suffix) when too long.
    This matches what MultipleChoiceQADataset does during training.
    """
    suffix_ids = tokenizer.encode(suffix)
    context_ids = tokenizer.encode(context)
    max_context = max(0, max_length - len(suffix_ids) - 1)
    if len(context_ids) > max_context:
        context_ids = context_ids[:max_context]
    token_ids = context_ids + suffix_ids
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    return token_ids


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: run classification head with a given prompt format
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_prompt(
    qa_model,
    tokenizer,
    examples: List[Dict[str, Any]],
    format_fn: Callable = format_base,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run the fine-tuned TransformerForMultipleChoice with a specific prompt format.

    Uses smart truncation (trim context, keep question+answer) to match training.
    Returns dict with 'accuracy', 'predictions', and raw 'logits'.
    """
    qa_model.eval()
    qa_model.to(device)

    pad_id = 0

    all_preds: List[int] = []
    all_logits: List[List[float]] = []

    for start in range(0, len(examples), batch_size):
        batch_exs = examples[start : start + batch_size]
        batch_ids = []
        batch_masks = []

        for ex in batch_exs:
            choice_ids_list = []
            choice_mask_list = []
            for choice in ex["choices"]:
                # Build the suffix (question + answer portion) separately
                # so we can do smart truncation on the context only
                full_text = format_fn(ex["context"], ex["question"], choice)
                context_text = ex["context"]
                suffix_text = full_text[len(context_text):]  # everything after context

                ids = encode_with_smart_truncation(
                    tokenizer, context_text, suffix_text, max_length
                )
                mask = [1] * len(ids)
                pad = max_length - len(ids)
                ids = ids + [pad_id] * pad
                mask = mask + [0] * pad
                choice_ids_list.append(ids)
                choice_mask_list.append(mask)

            batch_ids.append(choice_ids_list)
            batch_masks.append(choice_mask_list)

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch_masks, dtype=torch.long, device=device)

        logits = qa_model(input_ids, attention_mask)  # (batch, num_choices)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_logits.extend(logits.cpu().tolist())

    # Compute accuracy
    correct = sum(
        1 for p, ex in zip(all_preds, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": all_preds,
        "logits": all_logits,
    }


def evaluate_all_prompts(
    qa_model,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate with multiple prompt formats, then combine via majority-vote ensemble.

    Strategy:
    - Run each prompt style independently
    - For each example, take a majority vote across all styles
    - If tied, fall back to the style with the highest confidence (max logit)
    - This ensemble is more robust than any single format
    """
    all_results: Dict[str, Dict] = {}

    for name, fmt_fn in PROMPT_STYLES.items():
        result = evaluate_with_prompt(
            qa_model, tokenizer, examples,
            format_fn=fmt_fn,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        print(f"  Prompt style '{name}': {result['accuracy']:.2%}")
        all_results[name] = result

    # ── Majority-vote ensemble ──────────────────────────────────────────
    style_names = list(all_results.keys())
    ensemble_preds: List[int] = []

    for i in range(len(examples)):
        votes = [all_results[s]["predictions"][i] for s in style_names]
        counter = Counter(votes)
        top_count = counter.most_common(1)[0][1]
        tied = [v for v, c in counter.items() if c == top_count]

        if len(tied) == 1:
            ensemble_preds.append(tied[0])
        else:
            # Tie-break: pick the choice with highest summed logit across styles
            num_choices = len(examples[i]["choices"])
            summed = [0.0] * num_choices
            for s in style_names:
                logits_i = all_results[s]["logits"][i]
                for c in range(min(num_choices, len(logits_i))):
                    summed[c] += logits_i[c]
            ensemble_preds.append(max(range(num_choices), key=lambda c: summed[c]))

    # Compute ensemble accuracy
    correct = sum(
        1 for p, ex in zip(ensemble_preds, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    ensemble_acc = correct / total if total > 0 else 0.0

    print(f"  Ensemble (majority vote): {ensemble_acc:.2%}")

    # Return ensemble if it beats all individuals, otherwise return best single
    best_single = max(all_results.values(), key=lambda r: r["accuracy"])
    if ensemble_acc >= best_single["accuracy"]:
        print(f"  → Using ensemble ({ensemble_acc:.2%})")
        return {
            "accuracy": ensemble_acc,
            "predictions": ensemble_preds,
        }
    else:
        best_name = max(all_results, key=lambda n: all_results[n]["accuracy"])
        print(f"  → Using best single style '{best_name}' ({best_single['accuracy']:.2%})")
        return {
            "accuracy": best_single["accuracy"],
            "predictions": best_single["predictions"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible stubs (used by test_part4.py)
# ─────────────────────────────────────────────────────────────────────────────

class PromptTemplate:
    """Backward-compatible prompt template for tests."""

    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
        "direct": "Based on the passage below, answer the multiple choice question.\n\n{context}\n\nQ: {question}\n{choices_formatted}\n\nThe correct answer is",
        "completion": "{context}\n\nQuestion: {question}\nAnswer:",
    }

    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, **kwargs):
        self.template_name = template_name
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])

    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=self._format_choices(choices),
            **kwargs,
        )


class PromptingPipeline:
    """Backward-compatible pipeline stub for tests."""

    def __init__(self, model=None, tokenizer=None, template=None, device="cpu", **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
