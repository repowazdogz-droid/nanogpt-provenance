"""Tamper-evident, hash-chained provenance for a training run."""
from .canonical import canonical_bytes, sha256_hex, sha256_file, sha256_raw
from .chain import Chain, recompute_chain, seg_hash, GENESIS

__all__ = [
    "canonical_bytes", "sha256_hex", "sha256_file", "sha256_raw",
    "Chain", "recompute_chain", "seg_hash", "GENESIS",
]
