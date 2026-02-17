"""
Sampling utilities for text generation.
Example submission.
"""
import torch
from torch import Tensor
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


def greedy_decode(model, input_ids: Tensor, max_new_tokens: int, eos_token_id=None, pad_token_id=None) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
    return generated


def top_k_decode(model, input_ids: Tensor, max_new_tokens: int, k: int = 50, temperature: float = 1.0, eos_token_id=None, pad_token_id=None) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
            probs = softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_tokens = top_k_indices.gather(dim=-1, index=sampled_idx)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
    return generated


def nucleus_decode(model, input_ids: Tensor, max_new_tokens: int, p: float = 0.9, temperature: float = 1.0, eos_token_id=None, pad_token_id=None) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
            next_tokens = sorted_indices.gather(dim=-1, index=sampled_indices)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
    return generated


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, method: str = "greedy", k: int = 50, p: float = 0.9, temperature: float = 1.0, eos_token_id=None) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    if method == "greedy":
        output_ids = greedy_decode(model, input_ids, max_new_tokens, eos_token_id)
    elif method == "top_k":
        output_ids = top_k_decode(model, input_ids, max_new_tokens, k, temperature, eos_token_id)
    elif method == "nucleus":
        output_ids = nucleus_decode(model, input_ids, max_new_tokens, p, temperature, eos_token_id)
    else:
        raise ValueError(f"Unknown method: {method}")
    return tokenizer.decode(output_ids[0].tolist())
