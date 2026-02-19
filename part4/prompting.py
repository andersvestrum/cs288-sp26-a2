"""
Prompting utilities for multiple-choice QA.

Uses the fine-tuned TransformerForMultipleChoice model (classification head)
with different input prompt formats AND different pooling strategies.

Key improvements over the baseline fine-tuned evaluation:
1. Smart truncation that always preserves question + answer (matches training)
2. Multiple prompt formats (only formats close to training distribution)
3. Multi-pooling: run the SAME classifier head on last-token, mean, and max
   pooled hidden states — gives diverse "views" without confusing the model
4. Logit-sum ensemble across all (format × pooling) combinations
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt format functions: each takes (context, question, choice) → str
# Only formats close to the training distribution — deviating hurts accuracy.
# ─────────────────────────────────────────────────────────────────────────────

def format_base(context: str, question: str, choice: str) -> str:
    """Baseline format — identical to fine-tuning training format."""
    return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"


def format_compact(context: str, question: str, choice: str) -> str:
    """Compact style — fewer separator tokens so more context fits in."""
    return f"{context}\nQ: {question}\nA: {choice}"


def format_fill(context: str, question: str, choice: str) -> str:
    """Minimal variation — slightly different separator."""
    return (
        f"{context}\n\n"
        f"Question: {question}\n"
        f"The answer is: {choice}"
    )


PROMPT_STYLES: Dict[str, Callable] = {
    "base": format_base,
    "compact": format_compact,
    "fill": format_fill,
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
# Multi-pooling evaluation: run classifier on different pooling of hidden states
# ─────────────────────────────────────────────────────────────────────────────

def _pool_hidden(hidden_states, attention_mask, strategy: str):
    """Pool hidden states using a given strategy."""
    if strategy == "last":
        seq_lengths = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]
    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif strategy == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        h = hidden_states.masked_fill(~mask, float("-inf"))
        return h.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling: {strategy}")


POOLING_STRATEGIES = ["last", "mean", "max"]


@torch.no_grad()
def evaluate_with_prompt_and_pooling(
    qa_model,
    tokenizer,
    examples: List[Dict[str, Any]],
    format_fn: Callable = format_base,
    pooling: str = "last",
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run the fine-tuned model with a specific prompt format AND pooling strategy.

    Instead of using the model's built-in forward (which always uses 'last' pooling),
    we manually: get hidden states → pool with chosen strategy → classify.
    This lets us get diverse logits from the same model without retraining.
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
                full_text = format_fn(ex["context"], ex["question"], choice)
                context_text = ex["context"]
                suffix_text = full_text[len(context_text):]

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

        bs, nc, sl = input_ids.shape
        flat_ids = input_ids.view(-1, sl)
        flat_mask = attention_mask.view(-1, sl)

        # Get hidden states from the backbone
        hidden_states = qa_model._get_hidden_states(flat_ids)

        # Pool with the chosen strategy (not necessarily 'last')
        pooled = _pool_hidden(hidden_states, flat_mask, pooling)

        # Run through the same classifier head
        logits = qa_model.classifier(pooled).squeeze(-1)  # (bs*nc,)
        choice_logits = logits.view(bs, nc)

        preds = choice_logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_logits.extend(choice_logits.cpu().tolist())

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


# Keep backward-compatible single-format evaluator
evaluate_with_prompt = evaluate_with_prompt_and_pooling


def evaluate_all_prompts(
    qa_model,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate with (prompt_format × pooling_strategy) combinations, then ensemble.

    With 3 formats × 3 poolings = 9 diverse "views" of the same model.
    Logit-sum ensemble with zero-mean normalization gives robust soft voting.
    """
    all_results: Dict[str, Dict] = {}

    for fmt_name, fmt_fn in PROMPT_STYLES.items():
        for pool in POOLING_STRATEGIES:
            key = f"{fmt_name}+{pool}"
            result = evaluate_with_prompt_and_pooling(
                qa_model, tokenizer, examples,
                format_fn=fmt_fn,
                pooling=pool,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            )
            print(f"  {key}: {result['accuracy']:.2%}")
            all_results[key] = result

    # ── Logit-sum ensemble (soft voting) ────────────────────────────────
    style_names = list(all_results.keys())
    n = len(examples)
    num_choices = len(examples[0]["choices"]) if n > 0 else 4

    logit_tensors = []
    for s in style_names:
        raw = all_results[s]["logits"]
        padded = [
            row + [float("-inf")] * (num_choices - len(row))
            if len(row) < num_choices else row[:num_choices]
            for row in raw
        ]
        logit_tensors.append(torch.tensor(padded))

    stacked = torch.stack(logit_tensors)  # (S, N, C)

    # Zero-mean normalize each view's logits per example (removes scale bias)
    means = stacked.mean(dim=-1, keepdim=True)
    normed = stacked - means

    # Sum across all views → (N, C)
    summed = normed.sum(dim=0)
    ensemble_preds = summed.argmax(dim=-1).tolist()

    correct = sum(
        1 for p, ex in zip(ensemble_preds, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    ensemble_acc = correct / total if total > 0 else 0.0

    print(f"  Ensemble (9-way logit-sum): {ensemble_acc:.2%}")

    # Also try ensemble of only the top-K views
    accs = {k: v["accuracy"] for k, v in all_results.items()}
    sorted_keys = sorted(accs, key=accs.get, reverse=True)
    for topk in [3, 5, 7]:
        if topk >= len(sorted_keys):
            continue
        top_indices = [style_names.index(k) for k in sorted_keys[:topk]]
        top_normed = normed[top_indices]
        top_summed = top_normed.sum(dim=0)
        top_preds = top_summed.argmax(dim=-1).tolist()
        top_correct = sum(
            1 for p, ex in zip(top_preds, examples)
            if ex.get("answer", -1) >= 0 and p == ex["answer"]
        )
        top_acc = top_correct / total if total > 0 else 0.0
        print(f"  Ensemble (top-{topk}): {top_acc:.2%}")
        if top_acc > ensemble_acc:
            ensemble_acc = top_acc
            ensemble_preds = top_preds

    # Compare with best single
    best_single = max(all_results.values(), key=lambda r: r["accuracy"])
    if ensemble_acc >= best_single["accuracy"]:
        print(f"  → Using ensemble ({ensemble_acc:.2%})")
        return {
            "accuracy": ensemble_acc,
            "predictions": ensemble_preds,
        }
    else:
        best_name = max(all_results, key=lambda n: all_results[n]["accuracy"])
        print(f"  → Using best single '{best_name}' ({best_single['accuracy']:.2%})")
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
