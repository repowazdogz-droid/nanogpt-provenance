# SCOPE

What this artifact is, what it claims, and — precisely — where each claim stops.
Claims are tagged **PROVEN** / **MEASURED** / **OBSERVED** / **NOT PROVEN**.

## What it is

A tamper-evident, hash-chained **provenance layer** for a training run, plus an
**independent verifier** that reports one of three honest states — VERIFIED /
TAMPERED / UNKNOWN — with per-check evidence.

The training run is a tiny but faithful GPT (token + positional embeddings,
pre-LayerNorm causal multi-head self-attention, GELU MLP, AdamW, warmup+cosine
schedule) in pure NumPy, at a scale that runs on a laptop CPU in seconds. It is
**not** modded-nanoGPT and makes no speed claim. It exercises the same
*determinants of a result* — data, config, seed, code, environment, trajectory —
which is what the provenance layer is about.

## The question

> Can an ML training result be made independently verifiable — such that someone
> who does not trust the submitter can check the claim — and where does that
> verifiability honestly break down?

## The three states (what each means)

- **VERIFIED** — the chain recomputes, the recorded data reconstructs to the
  recorded hashes, the compute code matches this checkout, and **re-running
  training from the recorded config re-derives the recorded trajectory and final
  val loss bit-for-bit**, in an environment whose determinism-critical
  fingerprint matches.
- **TAMPERED** — the chain is broken, the recorded data hash ≠ the actual data,
  or the recorded numbers do **not** re-derive even though code and numeric
  environment match the record (internally inconsistent).
- **UNKNOWN** — the verifier cannot settle it: an input is not reconstructible,
  the recorded compute code differs from this checkout, or a re-derivation
  mismatch occurs under a *different* numeric environment than the record
  declares (mismatch cannot be attributed to tampering vs nondeterminism).
  **UNKNOWN is never silently upgraded to VERIFIED.**

## Claims

**PROVEN** (property of SHA-256 hash-chaining + demonstrated by negative controls):
- Any edit to a sealed segment payload that does **not** also recompute all
  downstream hashes and the seal breaks the chain and is caught
  (`chain_integrity`). Demonstrated by 4 naive-tamper controls.
- A *re-sealed* forgery (chain made internally consistent again) of the **final
  result** or a **logged loss** is caught by **re-derivation** in a matching
  environment; a re-sealed forgery of a **data hash** is caught by **data
  reconstruction**. Demonstrated by 3 reseal controls. Tamper-evidence therefore
  does not rest on the chain check alone — the semantic checks are load-bearing.
- The hand-written training backward pass agrees with finite-difference
  gradients (worst combined error 5.85e-07 over 120 sampled entries). The
  training engine is verified, not assumed. (`tests/gradcheck.py`)

**MEASURED** (on ONE machine — Apple M4 Max, macOS, Python 3.14.4, NumPy 2.4.4 on
Apple Accelerate BLAS — over a handful of runs this session):
- The run re-derives **bit-for-bit**. Final val loss =
  `0.10635284871660042` (`0x1.b39f0b7130106p-4`), identical across repeated runs
  and across **1, 8, and 16** BLAS thread counts (compute payloads identical;
  see `tests/determinism_probe.py`). So on this environment VERIFIED is genuine
  bit-exact re-derivation, not a rounded-number match.

**OBSERVED**:
- Two runs with different declared thread counts produce **identical compute
  payloads but different seals** — the seal binds the declared environment, so
  the same computation under a different declared environment is a different
  record. A human eyeballing the printed 6-decimal loss would call both
  "reproduced"; the full-precision seal distinguishes them.
- The **compute fingerprint** (hash of config+data+trajectory+result, excluding
  time/env) is **stable across independent captures**, while their **seals differ**
  because the seal binds capture time and environment. So the fingerprint is the
  reproducible identity two parties compare; the seal is a per-capture
  tamper-evidence tag (`tests/determinism_probe.py`).

**NOT PROVEN / out of scope**:
- **Bit-for-bit reproducibility across BLAS libraries.** Measured 2026-09-04
  (GAPS.md §3): a second Accelerate machine re-derives bit-for-bit; an x86-64
  OpenBLAS machine diverges at step 9 by 4.4e-16 and the verifier returns
  **UNKNOWN** for every cross-BLAS pair, not TAMPERED. Three CPU environments,
  one seed, one config, single-threaded. **A real GPU and multithreaded BLAS
  remain untested.**
- **That the code is correct** or is the intended algorithm. Re-derivation checks
  *reconstructibility of the result from the inputs*, not correctness. A run that
  faithfully re-derives a result produced by buggy code is still VERIFIED.
- **That the result is good.** Low loss on a short repetitive corpus is
  meaningless as a capability claim. Provenance ≠ quality.
- **Who ran it / accountability.** The record proves *what* was committed and
  that it was unaltered (integrity). It does not bind a submitter identity and
  does not establish responsibility. There are no signatures or attestation.
  (See GAPS.md and `provenance-establishes-integrity-not-accountability`.)
- **The runtime/compiled stack.** Python, NumPy, Accelerate, the OS, and the
  hardware are a trusted base, unverified here. See `TCB.md` for the full trusted
  base per verdict.

## Reproduce from clean

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh        # gradcheck, capture, verify (VERIFIED), 11 negative controls, determinism probe
```
Outputs stream to stdout and to `RESULTS.md`.
