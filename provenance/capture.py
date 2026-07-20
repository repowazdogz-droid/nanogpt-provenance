"""
Capture the environment that determines a run.

We record two categories, and the distinction matters for the verifier:

  DETERMINISM-CRITICAL fields (the "numeric fingerprint"): things that, if they
  differ, can change the produced floats bit-for-bit -- thread counts, dtype,
  numpy/BLAS build, platform/arch. The verifier treats a mismatch here as a
  reason it CANNOT claim VERIFIED (it returns UNKNOWN), because it can no longer
  expect bit-identical re-derivation.

  CONTEXTUAL fields: git commit, dirty flag, source hashes, timestamps. These
  establish provenance/integrity but a mismatch does not by itself make a
  numeric re-derivation impossible.

We hash SOURCE FILES by content (not just git commit) so the record does not
have to trust git -- a dirty or lying working tree is caught by the content hash.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys

import numpy as np

from .canonical import sha256_file

# COMPUTE_SOURCE: files whose content determines the TRAINING RESULT. A mismatch
# here means a re-run tests different code, so re-derivation can't be attributed
# (verdict UNKNOWN). These are the files the verifier must match to claim VERIFIED.
COMPUTE_SOURCE = ["model.py", "train.py", "data.py"]

# PROVENANCE_SOURCE: files that determine the RECORD FORMAT / hashing, not the
# numbers. Recorded for transparency, but a verifier is EXPECTED to run its own,
# independently-audited provenance library -- so a mismatch here does not block
# re-derivation (independent implementation is the point; see
# a-record-cannot-witness-itself).
PROVENANCE_SOURCE = ["provenance/canonical.py", "provenance/chain.py",
                     "provenance/capture.py"]

# Everything captured, for a complete content-addressed record of the tree.
SOURCE_FILES = COMPUTE_SOURCE + PROVENANCE_SOURCE


def _git(*args: str, root: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", root, *args],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def capture_code(root: str) -> dict:
    commit = _git("rev-parse", "HEAD", root=root)
    status = _git("status", "--porcelain", root=root)
    dirty = bool(status) if status is not None else None
    source_hashes = {}
    for rel in SOURCE_FILES:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            source_hashes[rel] = sha256_file(p)
        else:
            source_hashes[rel] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_hashes,  # content-addressed; does not trust git
    }


def capture_deps() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "numpy_version": np.__version__,
    }


def _thread_env() -> dict:
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
    return {k: os.environ.get(k) for k in keys}


def capture_env() -> dict:
    """Everything the verifier compares as the numeric fingerprint + context."""
    # numpy's BLAS build info -- Accelerate vs OpenBLAS produce different bits.
    try:
        blas = np.show_config(mode="dicts")  # numpy>=1.25
    except Exception:
        blas = None
    determinism_critical = {
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "float_dtype": "float64",  # the run pins float64 everywhere (see model.py)
        "thread_env": _thread_env(),
        "blas_backend": _blas_name(blas),
    }
    contextual = {
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "blas_config_present": blas is not None,
    }
    return {
        "determinism_critical": determinism_critical,
        "contextual": contextual,
    }


def _blas_name(blas) -> str | None:
    if not blas:
        return None
    try:
        for entry in blas.get("Build Dependencies", {}).get("blas", {}).values():
            return str(entry)
    except Exception:
        pass
    # fall back to a stable stringification of whatever we got
    try:
        deps = blas.get("Build Dependencies", {})
        b = deps.get("blas", {})
        return b.get("name") or b.get("found") or None
    except Exception:
        return None
