"""
Canonical serialization + hashing.

The whole provenance guarantee rests on ONE property: two parties who hold the
same logical object must compute the same bytes, hence the same hash. Ordinary
json.dumps does not give that -- float formatting and key order drift across
platforms. So we canonicalize explicitly:

  * dict keys are sorted
  * no insignificant whitespace
  * floats are encoded by their EXACT IEEE-754 bits (float.hex()), tagged, so
    0.1 hashes identically on every machine and no shortest-repr rounding can
    silently change the bytes.

canonical_bytes(obj) -> bytes           the bytes we hash
sha256_hex(obj)      -> str             hex digest of those bytes

This module is deliberately tiny and dependency-free so a third-party verifier
can re-implement it in an afternoon and confirm our hashes independently.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FLOAT_TAG = "__f64__"  # marks an exact float encoding


def _encode(obj: Any) -> Any:
    """Recursively rewrite obj into a JSON-safe, canonical-friendly shape.

    Floats become {"__f64__": "<hex>"} using float.hex(), which round-trips
    exactly for all finite float64 and has a stable spelling for inf/nan.
    """
    if isinstance(obj, bool):
        # bool is a subclass of int -- keep it as a JSON bool, not 0/1.
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {FLOAT_TAG: "nan"}
        if math.isinf(obj):
            return {FLOAT_TAG: "inf" if obj > 0 else "-inf"}
        return {FLOAT_TAG: obj.hex()}
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        # keys must be strings for JSON; enforce it rather than coerce silently.
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical dict keys must be str, got {type(k)}")
            out[k] = _encode(v)
        return out
    raise TypeError(f"cannot canonicalize object of type {type(obj)}")


def canonical_bytes(obj: Any) -> bytes:
    encoded = _encode(obj)
    text = json.dumps(
        encoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_raw(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
