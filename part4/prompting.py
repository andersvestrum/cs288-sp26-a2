"""
Prompting utilities for multiple-choice QA.

Uses the fine-tuned TransformerForMultipleChoice model (classification head)
with different input prompt formats. "Prompting" here means experimenting
with how context/question/answer are formatted before being fed to the
same classification model from Part 4A.
"""
import torch
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


def format_instruction(context: str, question: str, choice: str) -> str:
    """Prepend a short instruction."""
    return (
        f"Read the passage and answer the question.\n\n"
        f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
    )


def format_restated(context: str, question: str, choice: str) -> str:
    """Restate the question as part of the answer."""
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Based on the passage, the answer to \"{question}\" is: {choice}"
    )


def format_compact(context: str, question: str, choice: str) -> str:
    """Compact single-line style."""
    return f"{context}\nQ: {question}\nA: {choice}"


PROMPT_STYLES: Dict[str, Callable] = {
    "base": format_base,
    "instruction": format_instruction,
    "restated": format_restated,
    "compact": format_compact,
}


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

    For each example, each choice is formatted via format_fn(context, question, choice),
    tokenised, padded, and fed through the classification head. The choice with the
    highest logit wins.

    Returns dict with 'accuracy' and 'predictions' (ordered same as examples).
    """
    qa_model.eval()
    qa_model.to(device)

    pad_id = 0  # <|pad|> is token 0 or 1 depending on tokenizer; use 0 as safe default

    all_preds: List[int] = []

    for start in range(0, len(examples), batch_size):
        batch_exs = examples[start : start + batch_size]
        batch_ids = []
        batch_masks = []

        for ex in batch_exs:
            choice_ids_list = []
            choice_mask_list = []
            for choice in ex["choices"]:
                text = format_fn(ex["context"], ex["question"], choice)
                ids = tokenizer.encode(text)
                # Truncate from the RIGHT to match training truncation
                if len(ids) > max_length:
                    ids = ids[:max_length]
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
        preds = qa_model.predict(input_ids, attention_mask)
        all_preds.extend(preds.cpu().tolist())

    # Compute accuracy
    correct = sum(
        1 for p, ex in zip(all_preds, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": all_preds,
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
    Try all prompt styles and return the best result.
    """
    best_result = None
    best_name = None

    for name, fmt_fn in PROMPT_STYLES.items():
        result = evaluate_with_prompt(
            qa_model, tokenizer, examples,
            format_fn=fmt_fn,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        print(f"  Prompt style '{name}': {result['accuracy']:.2%}")
        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result
            best_name = name

    print(f"  Best prompt style: '{best_name}' ({best_result['accuracy']:.2%})")
    return best_result


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
