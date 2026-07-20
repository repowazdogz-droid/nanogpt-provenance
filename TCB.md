# TCB.md — Trusted Computing Base

What you must trust for each verdict to mean what it says. Nothing here is
established by the artifact itself; these are the foundations the evidence rests
on. The whole point of the three-state verifier is to keep this list honest: when
a needed assumption cannot be discharged, the verdict is UNKNOWN, not VERIFIED.

## For a VERIFIED verdict (re-derivation matched)

- **SHA-256 is collision-resistant.** The hash chain, the data/code content
  hashes, and the seal all rest on this. If it fails, tamper-evidence fails.
- **The verifier's own runtime**: CPython 3, NumPy, the OS, and the CPU. The
  verifier re-runs `run_training` on this stack; a VERIFIED result is only as
  trustworthy as the stack that reproduced it.
- **The verifier's numeric environment faithfully reproduces the record's.**
  VERIFIED rests on the *measured* fact that same-config + same-numeric-fingerprint
  re-derives bit-for-bit on this machine (Apple M4 Max, NumPy 2.4.4 on Apple
  Accelerate BLAS). The captured determinism-critical fingerprint (platform/arch,
  Python + NumPy versions, float dtype, BLAS backend, thread env) is a **best
  effort**, not a proof of sufficiency. This is the weakest link and it is exactly
  where the guarantee stops — see **GAPS.md §3**.
- **The compute code the verifier runs.** Re-derivation executes the *record's*
  `model.py` / `train.py` / `data.py` (matched by content hash). A VERIFIED result
  therefore establishes *reconstructibility of the result from the inputs*, NOT
  that this code is correct, and NOT that a second independent implementation
  would agree. The verifier does not cross-check against a reference model.

## For a TAMPERED verdict

- **SHA-256** (above), for chain and data-hash checks.
- **A matching numeric environment.** Tamper-detection by re-derivation only works
  when the verifier's fingerprint matches the record's. Under a foreign or
  nondeterministic environment, a genuine forgery degrades to UNKNOWN, not
  TAMPERED. Detection power degrades exactly where reproducibility does.

## What is NOT in the TCB (independent by design)

- **The record's claimed output.** The verifier never trusts the recorded loss; it
  recomputes it. This is the external reference of the
  `a-record-cannot-witness-itself` ladder.
- **The record author's provenance library.** A verifier is expected to run its
  own, independently-audited copy. `provenance/*.py` differences are reported but
  do not block re-derivation (only `model.py` / `train.py` / `data.py` do).
- **git.** Code state is captured by content hash, not by trusting git metadata; a
  dirty or lying working tree is caught by the content hash.

## What no verdict establishes

Integrity ≠ correctness ≠ quality ≠ accountability.

- **Not correctness.** A faithfully re-derived result produced by buggy code is
  still VERIFIED.
- **Not quality.** Low loss on a short repetitive corpus is not a capability claim.
- **Not accountability / identity.** The record proves *what* was committed and
  that it was unaltered, never *who* ran it. There are no identity signatures.
  (See `provenance-establishes-integrity-not-accountability`.)

## Scale and horizon

The demonstration is a 300-step NumPy GPT that re-derives bit-for-bit on one
deterministic machine. Cross-hardware re-derivation (a real GPU, a different BLAS)
is **untested and UNKNOWN** — not claimed, not implied achievable. The training
engine itself is trusted only as far as its finite-difference gradient check
(`tests/gradcheck.py`, worst combined error 5.85e-07) reaches.
