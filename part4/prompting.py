"""
Prompting utilities for multiple-choice QA.
Supports token scoring, choice log-likelihood scoring, and few-shot prompting.
"""
import torch
import math
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    """Prompt template with support for multiple formats and few-shot examples."""

    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
        "direct": "Based on the passage below, answer the multiple choice question.\n\n{context}\n\nQ: {question}\n{choices_formatted}\n\nThe correct answer is",
        "completion": "{context}\n\nQuestion: {question}\nAnswer:",
    }

    def __init__(
        self,
        template_name: str = "basic",
        custom_template: Optional[str] = None,
        choice_format: str = "letter",
        few_shot_data: Optional[List[Dict[str, Any]]] = None,
        num_shots: int = 0,
    ):
        self.template_name = template_name
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
        self.few_shot_data = few_shot_data or []
        self.num_shots = num_shots

    def _format_choices(self, choices: List[str]) -> str:
        labels = (
            ["A", "B", "C", "D", "E", "F", "G", "H"]
            if self.choice_format == "letter"
            else [str(i + 1) for i in range(len(choices))]
        )
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def _format_single(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=self._format_choices(choices),
            **kwargs,
        )

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        """Format a prompt, optionally prepending few-shot examples."""
        parts = []
        # Add few-shot demonstrations
        shots = self.few_shot_data[: self.num_shots] if self.num_shots > 0 else []
        for shot in shots:
            demo = self._format_single(shot["context"], shot["question"], shot["choices"])
            answer_idx = shot.get("answer", 0)
            label = chr(ord("A") + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
            parts.append(f"{demo} {label}")
        # Add the actual query
        parts.append(self._format_single(context, question, choices, **kwargs))
        return "\n\n".join(parts)

    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord("A") + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"

    def format_for_choice_ll(self, context: str, question: str, choice_text: str) -> str:
        """Format a prompt for choice log-likelihood scoring.
        
        Uses the SAME format as the fine-tuning dataset so the backbone's
        adapted representations transfer directly to prompting.
        """
        return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice_text}"


class PromptingPipeline:
    """
    Pipeline supporting two scoring methods:
      - 'token'     : predict the next-token letter (A/B/C/D) after the prompt
      - 'choice_ll' : score each choice by the average log-likelihood of its
                       answer text conditioned on the question+context prefix
    """

    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
        scoring_method: str = "token",
    ):
        self.model = model.to(device) if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.scoring_method = scoring_method
        self._setup_choice_tokens()

    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break

    # ------------------------------------------------------------------
    # Token-level scoring (predict A / B / C / D)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_token(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        token_ids = self.tokenizer.encode(prompt)
        # Truncate if needed (keep last context_length tokens)
        max_len = getattr(self.model, "context_length", 512)
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[:, -1, :]

        choice_labels = ["A", "B", "C", "D"][: len(choices)]
        choice_logits = []
        for label in choice_labels:
            if label in self.choice_tokens:
                choice_logits.append(logits[0, self.choice_tokens[label]].item())
            else:
                choice_logits.append(float("-inf"))

        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        prediction = probs.argmax().item()

        if return_probs:
            return prediction, probs.tolist()
        return prediction

    # ------------------------------------------------------------------
    # Choice log-likelihood scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_choice_ll(self, prefix_text: str, choice_text: str) -> float:
        """Return average log-likelihood of *choice_text* tokens given prefix."""
        self.model.eval()
        prefix_ids = self.tokenizer.encode(prefix_text)
        choice_ids = self.tokenizer.encode(" " + choice_text)  # space before answer

        if not choice_ids:
            return float("-inf")

        full_ids = prefix_ids + choice_ids
        max_len = getattr(self.model, "context_length", 512)
        if len(full_ids) > max_len:
            # trim prefix from the left to keep as much choice as possible
            trim = len(full_ids) - max_len
            full_ids = full_ids[trim:]
            # recompute where choice starts
            prefix_len = max(0, len(prefix_ids) - trim)
        else:
            prefix_len = len(prefix_ids)

        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids)  # (1, seq_len, vocab)

        # Compute log-likelihood over choice tokens
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total_ll = 0.0
        n_tokens = 0
        for i in range(prefix_len, len(full_ids)):
            target = full_ids[i]
            total_ll += log_probs[i - 1, target].item()
            n_tokens += 1

        return total_ll / max(n_tokens, 1)

    @torch.no_grad()
    def _predict_choice_ll(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        """Pick the choice whose answer text has highest avg log-likelihood.
        
        For each choice, we build the full string:
            "{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
        Then score only the *choice* tokens conditioned on the prefix.
        This matches the exact format used during fine-tuning so the
        backbone's adapted representations transfer directly.
        """
        self.model.eval()
        # Build the prefix that is shared across all choices
        prefix_text = f"{context}\n\nQuestion: {question}\n\nAnswer:"

        scores = []
        for choice in choices:
            ll = self._score_choice_ll(prefix_text, choice)
            scores.append(ll)

        scores_t = torch.tensor(scores)
        prediction = scores_t.argmax().item()
        if return_probs:
            probs = softmax(scores_t, dim=-1)
            return prediction, probs.tolist()
        return prediction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        if self.scoring_method == "choice_ll":
            return self._predict_choice_ll(context, question, choices, return_probs)
        else:
            return self._predict_token(context, question, choices, return_probs)

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(
        1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
