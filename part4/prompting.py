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


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
