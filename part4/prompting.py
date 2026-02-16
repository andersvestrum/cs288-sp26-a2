"""
Prompting utilities for multiple-choice QA.
Example submission.
"""
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
    }
    
    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), **kwargs)
    
    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class FewShotPromptTemplate(PromptTemplate):
    """Prompt template that includes few-shot examples before the test question."""
    
    def __init__(self, examples: List[Dict[str, Any]], num_shots: int = 2,
                 max_context_chars: int = 150, template_name: str = "simple", **kwargs):
        super().__init__(template_name=template_name, **kwargs)
        self.shot_examples = []
        for ex in examples:
            if len(self.shot_examples) >= num_shots:
                break
            if ex.get("answer", -1) >= 0:
                self.shot_examples.append(ex)
        self.max_context_chars = max_context_chars
    
    def _truncate_context(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    
    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        parts = []
        for ex in self.shot_examples:
            ctx = self._truncate_context(ex["context"], self.max_context_chars)
            shot_prompt = super().format(ctx, ex["question"], ex["choices"])
            answer_label = chr(ord('A') + ex["answer"])
            parts.append(f"{shot_prompt} {answer_label}")
        
        test_ctx = self._truncate_context(context, self.max_context_chars * 2)
        test_prompt = super().format(test_ctx, question, choices)
        parts.append(test_prompt)
        
        return "\n\n".join(parts)


class PromptingPipeline:
    def __init__(self, model, tokenizer, template: Optional[PromptTemplate] = None,
                 device: str = "cuda", max_length: int = 512):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.max_length = max_length
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[:, -1, :]
        
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
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


class LikelihoodPipeline:
    """Score each answer choice by how likely the model thinks it is as a continuation."""
    
    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 512):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
    
    @torch.no_grad()
    def _score_completion(self, prefix_ids: List[int], completion_ids: List[int]) -> float:
        """Compute average log-prob of completion_ids given prefix_ids."""
        if not completion_ids:
            return float("-inf")
        all_ids = prefix_ids + completion_ids
        if len(all_ids) > self.max_length:
            trim = len(all_ids) - self.max_length
            all_ids = all_ids[trim:]
            start = max(0, len(prefix_ids) - trim)
        else:
            start = len(prefix_ids)
        if start >= len(all_ids):
            return float("-inf")
        input_ids = torch.tensor([all_ids], device=self.device)
        logits = self.model(input_ids)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        count = 0
        for i in range(start, len(all_ids)):
            token_id = all_ids[i]
            total += log_probs[i - 1, token_id].item()
            count += 1
        return total / count if count > 0 else float("-inf")
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prefix = f"{context} {question} "
        prefix_ids = self.tokenizer.encode(prefix)
        scores = []
        for choice_text in choices:
            completion_ids = self.tokenizer.encode(choice_text)
            score = self._score_completion(prefix_ids, completion_ids)
            scores.append(score)
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction
    
    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


class FewShotLikelihoodPipeline(LikelihoodPipeline):
    """Few-shot version: prepend solved examples, then score each choice by likelihood."""
    
    def __init__(self, model, tokenizer, train_examples: List[Dict[str, Any]],
                 num_shots: int = 2, max_context_chars: int = 150,
                 device: str = "cuda", max_length: int = 512):
        super().__init__(model, tokenizer, device=device, max_length=max_length)
        self.shot_examples = []
        for ex in train_examples:
            if len(self.shot_examples) >= num_shots:
                break
            if ex.get("answer", -1) >= 0:
                self.shot_examples.append(ex)
        self.max_context_chars = max_context_chars
    
    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    
    def _build_shots_prefix(self) -> str:
        parts = []
        for ex in self.shot_examples:
            ctx = self._truncate(ex["context"], self.max_context_chars)
            answer_text = ex["choices"][ex["answer"]]
            parts.append(f"{ctx} {ex['question']} {answer_text}")
        return "\n\n".join(parts)
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        shots_prefix = self._build_shots_prefix()
        prefix = f"{shots_prefix}\n\n{context} {question} "
        prefix_ids = self.tokenizer.encode(prefix)
        scores = []
        for choice_text in choices:
            completion_ids = self.tokenizer.encode(choice_text)
            score = self._score_completion(prefix_ids, completion_ids)
            scores.append(score)
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction


class MatchedFormatLikelihoodPipeline:
    """Score each choice using the EXACT format from fine-tuning:
    {context}\\n\\nQuestion: {question}\\n\\nAnswer: {choice}

    Context is truncated (not question/answer) to fit within max_length.
    This matches MultipleChoiceQADataset._format_choice_input exactly.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 512):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def _prepare_ids(self, context: str, question: str, choice: str):
        """Tokenize with smart truncation: trim context to preserve Q&A."""
        qa_suffix = f"\n\nQuestion: {question}\n\nAnswer: "
        qa_suffix_ids = self.tokenizer.encode(qa_suffix)
        choice_ids = self.tokenizer.encode(choice)

        max_ctx = max(0, self.max_length - len(qa_suffix_ids) - len(choice_ids))
        context_ids = self.tokenizer.encode(context)
        if len(context_ids) > max_ctx:
            context_ids = context_ids[:max_ctx]

        full_ids = context_ids + qa_suffix_ids + choice_ids
        answer_start = len(context_ids) + len(qa_suffix_ids)
        return full_ids, answer_start

    @torch.no_grad()
    def _score_choice(self, context: str, question: str, choice: str) -> float:
        full_ids, answer_start = self._prepare_ids(context, question, choice)
        if answer_start >= len(full_ids):
            return float("-inf")

        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids)
        log_probs = torch.log_softmax(logits[0], dim=-1)

        total = 0.0
        count = 0
        for i in range(answer_start, len(full_ids)):
            total += log_probs[i - 1, full_ids[i]].item()
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str],
                       return_probs: bool = False):
        self.model.eval()
        scores = [self._score_choice(context, question, c) for c in choices]
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"])
                for ex in examples]


class CalibratedLikelihoodPipeline(MatchedFormatLikelihoodPipeline):
    """PMI-calibrated scoring: score = P(choice|ctx,q) - P(choice|neutral).
    Removes prior bias toward frequent tokens."""

    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 512,
                 calibration_context: str = "N/A"):
        super().__init__(model, tokenizer, device, max_length)
        self.cal_ctx = calibration_context
        self._prior_cache: Dict[str, float] = {}

    def _score_prior(self, choice: str) -> float:
        if choice not in self._prior_cache:
            self._prior_cache[choice] = self._score_choice(
                self.cal_ctx, self.cal_ctx, choice)
        return self._prior_cache[choice]

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str],
                       return_probs: bool = False):
        self.model.eval()
        scores = []
        for c in choices:
            actual = self._score_choice(context, question, c)
            prior = self._score_prior(c)
            scores.append(actual - prior)
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction


class FewShotMatchedLikelihoodPipeline:
    """Few-shot likelihood scoring with matched fine-tuning format.
    Prepends solved QA examples (in fine-tuning format), then scores each choice.
    Selects examples with shortest contexts to maximize token budget."""

    def __init__(self, model, tokenizer, train_examples: List[Dict[str, Any]],
                 num_shots: int = 2, device: str = "cuda", max_length: int = 512,
                 max_shot_tokens: int = 80):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        valid = [ex for ex in train_examples if ex.get("answer", -1) >= 0]
        valid.sort(key=lambda ex: len(ex.get("context", "")))
        self.shot_examples = valid[:num_shots]
        self.max_shot_tokens = max_shot_tokens
        self._shots_ids = self._build_shots()

    def _build_shots(self) -> List[int]:
        sep = self.tokenizer.encode("\n\n")
        all_ids: List[int] = []
        for i, ex in enumerate(self.shot_examples):
            if i > 0:
                all_ids.extend(sep)
            ctx_ids = self.tokenizer.encode(ex["context"])
            qa = f"\n\nQuestion: {ex['question']}\n\nAnswer: {ex['choices'][ex['answer']]}"
            qa_ids = self.tokenizer.encode(qa)
            budget = max(0, self.max_shot_tokens - len(qa_ids))
            if len(ctx_ids) > budget:
                ctx_ids = ctx_ids[:budget]
            all_ids.extend(ctx_ids + qa_ids)
        return all_ids

    @torch.no_grad()
    def _score_choice(self, context: str, question: str, choice: str) -> float:
        self.model.eval()
        sep = self.tokenizer.encode("\n\n")
        qa_suffix_ids = self.tokenizer.encode(f"\n\nQuestion: {question}\n\nAnswer: ")
        choice_ids = self.tokenizer.encode(choice)

        fixed = len(self._shots_ids) + len(sep) + len(qa_suffix_ids) + len(choice_ids)
        max_ctx = max(0, self.max_length - fixed)
        ctx_ids = self.tokenizer.encode(context)
        if len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[:max_ctx]

        full_ids = self._shots_ids + sep + ctx_ids + qa_suffix_ids + choice_ids
        answer_start = len(full_ids) - len(choice_ids)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
        if answer_start >= len(full_ids) or answer_start < 1:
            return float("-inf")

        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids)
        log_probs = torch.log_softmax(logits[0], dim=-1)

        total = 0.0
        count = 0
        for i in range(answer_start, len(full_ids)):
            total += log_probs[i - 1, full_ids[i]].item()
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str],
                       return_probs: bool = False):
        scores = [self._score_choice(context, question, c) for c in choices]
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"])
                for ex in examples]


class CalibratedFewShotMatchedPipeline:
    """Combines few-shot matched format with PMI calibration.
    Uses few-shot examples for context, then calibrates to remove prior bias."""

    def __init__(self, model, tokenizer, train_examples: List[Dict[str, Any]],
                 num_shots: int = 2, device: str = "cuda", max_length: int = 512,
                 max_shot_tokens: int = 80, calibration_context: str = "N/A"):
        self.fewshot = FewShotMatchedLikelihoodPipeline(
            model, tokenizer, train_examples, num_shots, device, max_length, max_shot_tokens)
        self.matched = MatchedFormatLikelihoodPipeline(model, tokenizer, device, max_length)
        self.cal_ctx = calibration_context
        self._prior_cache: Dict[str, float] = {}
        self.model = self.fewshot.model
        self.tokenizer = tokenizer
        self.device = device

    def _score_prior(self, choice: str) -> float:
        if choice not in self._prior_cache:
            self._prior_cache[choice] = self.matched._score_choice(
                self.cal_ctx, self.cal_ctx, choice)
        return self._prior_cache[choice]

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str],
                       return_probs: bool = False):
        self.model.eval()
        scores = []
        for c in choices:
            actual = self.fewshot._score_choice(context, question, c)
            prior = self._score_prior(c)
            scores.append(actual - prior)
        scores_t = torch.tensor(scores)
        probs = softmax(scores_t, dim=-1)
        prediction = probs.argmax().item()
        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"])
                for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
