"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator


# GPT-2 pre-tokenization pattern
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.
    
    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []
    
    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return
    
    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Build a pattern that matches special tokens
    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"
    
    # Split text by special tokens
    parts = std_re.split(split_pattern, text)
    
    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization
            for match in GPT2_PAT.finditer(part):
                yield match.group()


def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include (e.g., ["<|endoftext|>"])
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs in order they were learned [(bytes, bytes), ...]
    
    Algorithm Overview:
        BPE iteratively merges the most frequent pair of adjacent tokens until
        the vocabulary reaches the target size.
    
    Detailed Steps:
    
    1. VOCABULARY INITIALIZATION
       The initial vocabulary is built in this exact order:
       - First: Add special tokens (in the order provided)
       - Then: Add all 256 single-byte values (0x00 to 0xFF)
       
       Example with special_tokens=["<|endoftext|>"]:
         vocab = {
             0: b"<|endoftext|>",   # Special token first
             1: b"\\x00",           # Byte 0
             2: b"\\x01",           # Byte 1
             ...
             256: b"\\xff",         # Byte 255
         }
       
       So the initial vocab size = len(special_tokens) + 256
    
    2. WORD FREQUENCY COUNTING
       - Pre-tokenize the corpus using pre_tokenize(text, special_tokens)
       - For each pre-token, convert to bytes and represent as tuple of single bytes
       - Skip any word containing a "forbidden substring" (prefix of a special token)
       
       Example: "hello" -> (b'h', b'e', b'l', b'l', b'o')
       
       word_freqs is a Counter mapping: tuple[bytes, ...] -> frequency
    
    3. PAIR FREQUENCY COUNTING  
       Count how often each adjacent pair appears across ALL words, weighted by
       word frequency.
       
       Example: If word (b'h', b'e', b'l', b'l', b'o') appears 10 times:
         - pair (b'h', b'e') gets +10
         - pair (b'e', b'l') gets +10
         - pair (b'l', b'l') gets +10
         - pair (b'l', b'o') gets +10
    
    4. MERGE LOOP (repeat until vocab_size is reached)
       
       a. SELECT BEST PAIR (DETERMINISTIC TIE-BREAKING):
          Find the pair with highest frequency. If multiple pairs have the same
          frequency, select the lexicographically smallest pair.
          
          Lexicographic comparison on (bytes, bytes) tuples:
            - Compare first element as bytes
            - If equal, compare second element as bytes
          
          Example: If pairs (b'a', b'b') and (b'a', b'c') both have freq=100,
                   select (b'a', b'b') because b'b' < b'c'
          
          Implementation: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          This sorts by (frequency, pair) and takes the max.
                          Since we want highest freq but lowest pair for ties,
                          use: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          
                          Note: Python compares bytes lexicographically by default.
       
       b. CREATE MERGED TOKEN:
          new_token = first + second  (bytes concatenation)
          Add to vocabulary with next available token_id
          Append (first, second) to merges list
       
       c. UPDATE WORD REPRESENTATIONS:
          For each word in word_freqs, apply the merge using merge_word()
          This replaces all occurrences of the pair with the merged token
       
       d. UPDATE PAIR COUNTS:
          Recompute pair frequencies for the updated words
          (Or incrementally update - subtract old pairs, add new pairs)
    
    5. RETURN
       Return (vocab, merges) where merges is the list of pairs in the order
       they were merged.
    
    Performance Note:
        A naive implementation recomputing all pair counts each iteration is O(n²).
        For efficiency, incrementally update pair counts by only processing words
        that contained the merged pair.
    """
    special_tokens = special_tokens or []
    
    # Read the corpus
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    # Build set of "forbidden" substrings from special tokens
    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])
    
    # TODO: Implement BPE training
    # 1. Initialize vocabulary
    vocab = {}
    token_id = 0
    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    # 2. Word frequency counting
    word_freqs = Counter()
    for pre_token in pre_tokenize(text, special_tokens):
        byte_seq = pre_token.encode("utf-8")

        if any(fs in byte_seq for fs in forbidden_substrings):
            continue

        word = tuple(bytes([b]) for b in byte_seq)
        word_freqs[word] += 1

    # 3. Initial pair frequency counting
    pair_counts = Counter()
    pair_to_words = {}
    for word, freq in word_freqs.items():
        for pair in get_pairs(word):
            pair_counts[pair] += freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)


    # 4. Merge loop (repeated until vocab_size is reached)
    merges = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Select best pair (highest freq, then lexicographically smallest for ties)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        # Create merged token
        new_token = best_pair[0] + best_pair[1]
        vocab[token_id] = new_token
        token_id += 1
        merges.append(best_pair)

        # Incrementally update: only process words that contain the merged pair
        affected_words = pair_to_words.get(best_pair, set()).copy()

        for old_word in affected_words:
            if old_word not in word_freqs:
                continue
            freq = word_freqs[old_word]

            new_word = merge_word(old_word, best_pair)
            if new_word == old_word:
                continue

            # Remove old word's pair contributions
            for pair in get_pairs(old_word):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(old_word)

            # Replace old word with new word in word_freqs
            del word_freqs[old_word]
            word_freqs[new_word] += freq

            # Add new word's pair contributions
            for pair in get_pairs(new_word):
                pair_counts[pair] += freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(new_word)

    return vocab, merges