"""
Prompting utilities for multiple-choice QA.

Provides:
  - PromptTemplate: Zero-shot prompt formatting (basic/instruction/simple)
  - FewShotPromptTemplate: Few-shot prompt formatting with in-context examples
  - PromptingPipeline: Next-token (A/B/C/D) prediction pipeline
  - PerplexityScoringPipeline: Scores each choice by log-likelihood
  - evaluate_prompting: Evaluation helper
"""
import random
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptTemplate:
    """Zero-shot prompt template for multiple-choice QA."""

    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
    }

    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format

    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i + 1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), **kwargs)

    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class FewShotPromptTemplate(PromptTemplate):
    """
    Few-shot prompt template that prepends solved examples before the test question.

    The in-context examples teach the model the expected QA pattern:
        [Example 1 context + question + choices -> answer letter]
        [Example 2 context + question + choices -> answer letter]
        [Test context + question + choices -> ???]

    Args:
        n_examples: Number of few-shot examples to include.
        train_examples: List of training examples to draw from.
        max_context_words: Maximum words to keep from each example context (truncation).
        seed: Random seed for reproducible example selection.
        choice_format: "letter" for A/B/C/D or "number" for 1/2/3/4.
    """

    def __init__(
        self,
        n_examples: int = 3,
        train_examples: Optional[List[Dict[str, Any]]] = None,
        max_context_words: int = 40,
        seed: int = 42,
        choice_format: str = "letter",
    ):
        super().__init__(template_name="basic", choice_format=choice_format)
        self.n_examples = n_examples
        self.train_examples = train_examples or []
        self.max_context_words = max_context_words
        self.seed = seed
        self._selected_examples: Optional[List[Dict]] = None

    def _truncate_text(self, text: str, max_words: int) -> str:
        """Truncate text to max_words, adding ellipsis if truncated."""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text

    def _select_examples(self) -> List[Dict[str, Any]]:
        """Select few-shot examples (cached after first call for consistency)."""
        if self._selected_examples is not None:
            return self._selected_examples
        if not self.train_examples:
            self._selected_examples = []
            return self._selected_examples

        rng = random.Random(self.seed)
        # Prefer examples with shorter contexts so they fit within token budget
        sorted_by_length = sorted(self.train_examples, key=lambda x: len(x.get("context", "")))
        pool_size = min(len(sorted_by_length), self.n_examples * 20)
        pool = sorted_by_length[:pool_size]
        n = min(self.n_examples, len(pool))
        self._selected_examples = rng.sample(pool, n)
        return self._selected_examples

    def _format_solved_example(self, example: Dict[str, Any]) -> str:
        """Format a single solved example as: context -> question -> choices -> answer."""
        ctx = self._truncate_text(example["context"], self.max_context_words)
        question = example["question"]
        choices = example["choices"]
        answer_label = chr(ord('A') + example["answer"])
        choices_text = self._format_choices(choices)
        return f"Context: {ctx}\nQuestion: {question}\n{choices_text}\nAnswer: {answer_label}"

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        """Build the full few-shot prompt: examples + test question."""
        examples = self._select_examples()
        parts = [self._format_solved_example(ex) for ex in examples]

        choices_text = self._format_choices(choices)
        parts.append(f"Context: {context}\nQuestion: {question}\n{choices_text}\nAnswer:")
        return "\n\n".join(parts)

    def reset_cache(self):
        """Clear cached examples so next format() re-selects."""
        self._selected_examples = None


# =============================================================================
# Prompting Pipelines
# =============================================================================

class PromptingPipeline:
    """
    Zero-shot / few-shot prompting pipeline using next-token prediction.

    For each QA example the pipeline:
      1. Formats the prompt using the template.
      2. Feeds the prompt through the LM.
      3. Reads logits at the last position for tokens A, B, C, D.
      4. Predicts the choice with the highest probability.
    """

    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.max_length = max_length
        self._setup_choice_tokens()

    def _setup_choice_tokens(self):
        """Find token IDs that correspond to answer letters A, B, C, D."""
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        """Predict the answer for a single QA example."""
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        input_ids = self.tokenizer.encode(prompt)

        # Truncate from the beginning to fit context_length (keep the end = test question)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]

        input_tensor = torch.tensor([input_ids], device=self.device)
        logits = self.model(input_tensor)[:, -1, :]

        choice_labels = ["A", "B", "C", "D"][:len(choices)]
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

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        """Predict answers for a batch of QA examples (processed sequentially)."""
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


class PerplexityScoringPipeline:
    """
    Alternative prompting approach: score each choice by log-likelihood.

    Instead of predicting A/B/C/D tokens, this pipeline:
      1. For each choice, creates text: "context question answer: <choice>"
      2. Computes the average log-probability of the choice tokens.
      3. Selects the choice with the highest average log-probability.

    This can outperform token-prediction when the model hasn't learned
    the A/B/C/D mapping, since it leverages raw language modeling ability.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 512):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_scores: bool = False):
        """Score each choice by log-likelihood and return the best."""
        self.model.eval()
        scores = []

        for choice in choices:
            prefix_text = f"{context}\n\nQuestion: {question}\nAnswer: "
            full_text = prefix_text + choice

            prefix_ids = self.tokenizer.encode(prefix_text)
            full_ids = self.tokenizer.encode(full_text)
            choice_start = len(prefix_ids)

            # Truncate prefix from beginning if needed
            if len(full_ids) > self.max_length:
                excess = len(full_ids) - self.max_length
                full_ids = full_ids[excess:]
                choice_start = max(0, choice_start - excess)

            if choice_start >= len(full_ids) or choice_start < 1:
                scores.append(float("-inf"))
                continue

            input_tensor = torch.tensor([full_ids], device=self.device)
            logits = self.model(input_tensor)

            # Compute average log-prob of choice tokens
            log_probs = torch.log_softmax(logits[0], dim=-1)
            token_log_probs = []
            for i in range(choice_start, len(full_ids)):
                token_id = full_ids[i]
                # logits at position i-1 predict token at position i
                token_log_probs.append(log_probs[i - 1, token_id].item())

            score = sum(token_log_probs) / len(token_log_probs) if token_log_probs else float("-inf")
            scores.append(score)

        prediction = scores.index(max(scores))
        if return_scores:
            return prediction, scores
        return prediction

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


class EnsemblePipeline:
    """
    Ensemble of prompting + perplexity scoring for improved accuracy.

    Combines predictions from a PromptingPipeline and a PerplexityScoringPipeline
    by averaging their probability/score distributions.
    """

    def __init__(self, prompt_pipeline: PromptingPipeline, ppl_pipeline: PerplexityScoringPipeline, weight_prompt: float = 0.5):
        self.prompt_pipeline = prompt_pipeline
        self.ppl_pipeline = ppl_pipeline
        self.weight_prompt = weight_prompt

    def predict_single(self, context: str, question: str, choices: List[str]):
        _, probs = self.prompt_pipeline.predict_single(context, question, choices, return_probs=True)
        _, scores = self.ppl_pipeline.predict_single(context, question, choices, return_scores=True)

        # Normalise perplexity scores to probabilities
        score_tensor = torch.tensor(scores)
        score_probs = softmax(score_tensor, dim=-1).tolist()

        # Weighted combination
        combined = [self.weight_prompt * p + (1 - self.weight_prompt) * s for p, s in zip(probs, score_probs)]
        return combined.index(max(combined))

    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    """Evaluate a prompting pipeline on QA examples and return accuracy + predictions."""
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
