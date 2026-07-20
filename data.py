"""
Tiny, fully-embedded char-level dataset.

The corpus is embedded as a literal string so a run reproduces from a clean clone
with no download (the data hash is therefore stable and submitter-independent).
This mirrors the nanoGPT data step -- tokenize, split train/val -- at toy scale.

build_dataset() returns a Dataset with:
  train, val   : int arrays of token ids
  vocab        : sorted list of chars
  source_bytes : the exact bytes we tokenized (so we can hash the *source*, not
                 just the derived token array)
"""
from __future__ import annotations

import numpy as np

# A short, self-authored public-domain-safe corpus with enough structure that a
# tiny model measurably lowers loss. Deterministic and embedded on purpose.
CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a provenance record proves what was committed and that it was not altered. "
    "it does not prove that the result is good, only that the inputs match the claim. "
    "verify the chain, re-derive the loss, and report unknown when you cannot. "
) * 12


class Dataset:
    def __init__(self, train, val, vocab, source_bytes):
        self.train = train
        self.val = val
        self.vocab = vocab
        self.source_bytes = source_bytes
        self.vocab_size = len(vocab)


def build_dataset(val_frac: float = 0.1) -> Dataset:
    source_bytes = CORPUS.encode("utf-8")
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in CORPUS], dtype=np.int64)
    n_val = int(len(ids) * val_frac)
    # val is the TAIL of the stream (deterministic split, no shuffling).
    train = ids[:-n_val]
    val = ids[-n_val:]
    return Dataset(train, val, chars, source_bytes)


def get_batch(data: np.ndarray, block_size: int, batch_size: int,
              rng: np.random.Generator):
    """Sample a batch. Deterministic given rng -- the verifier reconstructs the
    same rng state from the seed, so it draws the identical batches."""
    max_start = len(data) - block_size - 1
    ix = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y
