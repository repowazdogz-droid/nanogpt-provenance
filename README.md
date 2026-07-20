# nanogpt-provenance

A **tamper-evident provenance + re-derivation mechanism** for a training run, with
an **independent verifier** that RECOMPUTES the claimed result from the recorded
inputs rather than trusting it, and reports **VERIFIED / TAMPERED / UNKNOWN** with
evidence per finding.

This repository is a **mechanism artifact**, not a production service and not a
provenance system for real (large, GPU) training runs. It demonstrates the
mechanism on a deliberately **deterministic toy run** (a 300-step, gradient-check-
validated NumPy GPT) and **identifies exactly where the guarantee stops**. Every
claim is tagged by evidence grade (see [`SCOPE.md`](SCOPE.md)).

## What it is (the working part)

The NanoGPT speedrun culture posts "reproducible logs," but a reproducible log
today means "here is my output." There is no tamper-evident, independently
checkable provenance a third party can verify without trusting the submitter.
This is a working probe of that gap:

- The verifier **re-runs training from the recorded config and recomputes the
  claimed loss.** It never trusts the recorded number; a VERIFIED verdict means
  the result *re-derived bit-for-bit*.
- It **catches re-sealed forgeries** — an attacker who edits a logged loss or the
  final result and then recomputes *all* downstream hashes and the seal so the
  chain is internally consistent again. The naive hash-chain check passes such a
  forgery; the **semantic re-derivation and data-reconstruction checks catch it.**
  Tamper-evidence here does not rest on the chain alone.
- It is **adversarially tested**: a 12-case negative-control suite where each
  tampering must be caught, and each of the three verdict states is exercised by a
  real trigger (`tests/negative_controls.py`).

## The load-bearing limitation (read this before anything else)

**Re-derivation-based verification requires a matching numeric environment.** The
verifier can only confirm a result by reproducing its exact bits, and floating-
point results depend on the hardware and math library that produced them. So:

- Under a **foreign or nondeterministic environment**, a re-derivation mismatch
  cannot be attributed to tampering versus nondeterminism, and the verdict is
  **UNKNOWN** — *even for a genuine forgery*. **Tamper-detection power degrades
  exactly where reproducibility does** (demonstrated by the "forged loss under
  foreign env" control).
- The **real-world GPU case is precisely this nondeterministic case.** GPU kernels,
  and multithreaded BLAS libraries generally, do not guarantee bit-identical
  reductions run to run or across hardware.
- Therefore the mechanism is demonstrated on a **deterministic toy run**. On this
  machine NumPy is backed by Apple Accelerate, which turned out to be bit-identical
  across 1/8/16 threads, so the run re-derives exactly and VERIFIED is a genuine
  recomputation. **Cross-hardware re-derivation is UNTESTED and UNKNOWN here** — I
  have one machine and one BLAS. It is **not claimed, and not implied achievable.**

That boundary is the contribution, not a failure. See [`GAPS.md`](GAPS.md)
(§3 especially) and [`TCB.md`](TCB.md).

## Integrity ≠ correctness ≠ quality ≠ accountability

A VERIFIED verdict establishes **reconstructibility of the result from the inputs**
and nothing more. In the `a-record-cannot-witness-itself` ladder it reaches
WITNESSED-on-a-matching-environment. It does **not** establish:

- **correctness** — a faithfully re-derived result from buggy code is still VERIFIED;
- **quality** — low loss on a short repetitive corpus is not a capability claim;
- **accountability / identity** — the record proves *what* was committed and that it
  was unaltered, never *who* ran it. There are no signatures.
  (See `provenance-establishes-integrity-not-accountability`.)

## The blind spot I found by probing past my own green suite

The negative-control suite passed 12/12. That is exactly when a check is most
trusted and least examined, so I probed further, adversarially, past it. The probe
found a real hole: the `meta` block (capture timestamp, tool name) originally sat
**outside** the seal, so it could be edited and the record still verified as
VERIFIED. An unsealed-but-authoritative timestamp can mislead.

It is now folded into the hash chain (tamper-evident; the verifier guarantees the
timestamp is *unaltered*, never that it is *true* — integrity, not truth) and
guarded by a **named regression test**: the `naive: edit sealed meta timestamp`
control in `tests/negative_controls.py`. Documenting this as a first-class part of
the story is deliberate: the discipline of probing past a green check is the point,
not an embarrassment to bury.

## What gets captured (the determinants of a result)

| segment | contents |
|---|---|
| `config` | all hyperparameters, seed, schedule, model shape |
| `data` | sha256 of exact train / val tokens + the source corpus, vocab, counts |
| `code` | git commit + dirty flag, and **content hashes** of the compute source (does not trust git) |
| `env` | platform/arch, Python + NumPy versions, float dtype, BLAS backend, thread config (the *determinism-critical fingerprint*) |
| `meta` | capture time, tool (sealed: tamper-evident, not asserted true) |
| `trajectory` | per-step loss + LR, periodic val loss |
| `result` | final val/train loss — the **claim**, bound into the seal |

Segments are SHA-256 hash-chained: altering any payload changes every downstream
hash and the final `seal` no longer matches.

Two different identities, deliberately: the **seal** binds the *whole* record
including capture time and environment, so it is a per-capture tamper-evidence tag
(re-capturing gives a new seal). The **compute fingerprint** (a hash of
`config`+`data`+`trajectory`+`result`, excluding time/env) is the reproducible,
environment-independent identity of the computation — what two parties compare to
see if they ran the same thing. Verified stable across independent captures while
seals differ (`tests/determinism_probe.py`).

## Quickstart

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs, in order: the gradient check (the training engine is verified,
not assumed), a capture, a verify (VERIFIED), the 12 negative controls, and the
determinism probe. Output is also written to `RESULTS.md`.

Individually:
```
python3 train.py  --out runs/record.json      # capture a run
python3 verify.py runs/record.json            # VERIFIED / TAMPERED / UNKNOWN (+ evidence)
python3 tests/negative_controls.py            # each tampering must be caught
python3 tests/gradcheck.py                    # finite-difference gradient check
python3 tests/determinism_probe.py            # re-derivation across thread counts
```

Pin BLAS threads before running for bit-exact re-derivation:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`
(`run_all.sh` does this for you).

## How the verifier decides

1. **chain_integrity** — recompute every segment hash + seal. Broken ⇒ TAMPERED.
2. **data_reconstruction** — rebuild the dataset, recompute hashes vs the record.
   Mismatch ⇒ TAMPERED.
3. **code_integrity** — recompute compute-source (`model/train/data`) hashes vs
   this checkout. Mismatch ⇒ can't attribute a re-run ⇒ UNKNOWN. (Provenance-lib
   differences are reported but not disqualifying: an independent verifier is
   *expected* to run its own provenance code.)
4. **environment_fingerprint** — compare determinism-critical fields to now.
5. **re_derivation** — re-run `run_training(config)` and compare the trajectory
   and final metrics **bit-for-bit**.
   - match + matching env ⇒ **VERIFIED**
   - mismatch + matching env + matching code ⇒ **TAMPERED** (internally inconsistent)
   - mismatch + *different* env ⇒ **UNKNOWN** (cannot attribute)

## Where this sits in the portfolio

This extends the reconstructibility / tamper-evidence thread — the OMEGA
hash-chained provenance and `a-record-cannot-witness-itself` work — from decision
and audit records into the **ML-training-provenance** domain. The finding it
carries into that domain is a boundary, and the boundary is the contribution:
**re-derivation-based verification needs a matching numeric environment**, so it
gives strong tamper-evidence and genuine reconstructibility on a deterministic run
while cross-hardware re-derivation stays honestly UNKNOWN. Same discipline as the
sibling artifacts (`escrow-budget`, `capctl-iris`, `vsf-cjson`): claim exactly what
the evidence grade supports, and make the limit lead.

## Layout

```
model.py                 tiny GPT (manual forward/backward), AdamW, LR schedule
data.py                  embedded char-level corpus, deterministic batching
train.py                 run_training() [pure fn] + capture_and_seal() -> record
verify.py                the independent verifier (VERIFIED/TAMPERED/UNKNOWN)
provenance/
  canonical.py           canonical serialization + SHA-256 (floats by exact bits)
  chain.py               hash chain, recompute_chain(), reseal() [adversary tool]
  capture.py             code / deps / environment fingerprinting
tests/
  gradcheck.py           finite-difference check of the backward pass
  negative_controls.py   12 tampering cases; each must be caught
  determinism_probe.py   re-derivation across BLAS thread counts
SCOPE.md  GAPS.md  TCB.md   the honesty documents — read these
```

## Files to read first

- [`SCOPE.md`](SCOPE.md) — claims tagged PROVEN / MEASURED / OBSERVED / NOT PROVEN.
- [`GAPS.md`](GAPS.md) — what this establishes vs what it does not.
- [`TCB.md`](TCB.md) — the trusted computing base per verdict.

The mechanism is the easy part; the boundary is the contribution.
