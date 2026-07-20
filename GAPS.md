# GAPS — what this establishes vs what it does not

The contribution is the honest boundary, not a reproducibility miracle. This
document is deliberately the most important file in the repo.

## 1. Integrity is not accountability

The record proves **what** was committed and that it was **not altered**
(integrity). It does **not** prove **who** is responsible, or that the submitter
is who they claim. There are no identity signatures. A tamper-evident record with
a perfect seal still says nothing about accountability — these are different
properties and the strength of the seal must not be read as if it answered "who".
(See `provenance-establishes-integrity-not-accountability`.)

*How to close it (not done here):* sign the seal with a submitter key and publish
the record to an append-only transparency log (e.g. a Merkle log with inclusion
proofs) so the binding to an identity and a time is itself independently
checkable. That adds **attribution**, still not **correctness**.

## 2. A record cannot witness itself — what the verifier actually brings

Reproducing the claimed loss needs something the verifier holds that did **not**
come from the record's *claimed output*: here that is an **independent
re-execution** of the content-addressed compute code (`run_training`) on the
verifier's own machine. The verifier never trusts the recorded final loss — it
recomputes it. In the `a-record-cannot-witness-itself` ladder a clean result
therefore reaches **WITNESSED-on-a-matching-environment**: the result is
faithfully reconstructible from the inputs.

The honesty boundary, verbatim: **this proves reconstructibility of the result
from the inputs, NOT that the result is correct or good.** And the independence
is only from the *claimed output* — not from the hardware or runtime. The
verifier re-runs the *same* code; it does not cross-check against a second,
independent implementation of the model. A hidden nondeterminism in that code
that happens to be stable on this machine would be reproduced, not caught.

## 3. The GPU / cross-hardware boundary is UNKNOWN — and untested here

The prompt anticipated GPU nondeterminism. What actually happened on this machine:

- NumPy is backed by **Apple Accelerate** BLAS. Matmuls (incl. 1024×1024) are
  **bit-identical across 1/8/16 threads and both float32/float64**. So I could
  **not** manufacture same-machine numeric nondeterminism via threading.
- Consequently the training run re-derives **bit-for-bit** on this environment,
  and VERIFIED is a genuine bit-exact re-derivation.

This means the real nondeterminism boundary — a different GPU, a different BLAS
(e.g. multithreaded OpenBLAS with run-to-run reduction-order variation),
different hardware or a different math library — is **NOT PROVEN and untested
here**: I have one machine and one BLAS. The literature is clear that such
differences change low-order bits; I did not reproduce it, so I do not claim it,
and I equally do not claim reproduction is *achievable* across hardware.

The verifier's design is honest about this: when its numeric-environment
fingerprint differs from the record's, a re-derivation **mismatch** yields
**UNKNOWN**, not TAMPERED — the mismatch cannot be attributed to tampering vs
environment. A key consequence: **tamper-detection by re-derivation requires a
matching numeric environment.** Under an unverifiable/foreign environment, even a
genuine forgery of the loss can only be reported as UNKNOWN (demonstrated by the
"forged loss under foreign env" control). Detection power degrades exactly where
reproducibility does.

## 4. What "same environment" is trusted to mean

VERIFIED rests on the empirical fact that same-config + same-numeric-environment
re-derives identically on this machine (measured, N small, one session). The
determinism-critical fingerprint captured is: platform/arch, Python + NumPy
versions, float dtype, BLAS backend, and thread env. This list is a **best
effort**, not a proof of sufficiency — there could be an uncaptured factor that
matters on some system. If such a factor differs and the numbers diverge, the
verifier lands on UNKNOWN (it cannot attribute), which is the safe direction. It
would only be *wrong* if an uncaptured factor changed the numbers while the
captured fingerprint still matched — possible in principle, not observed here.

## 5. Toy scale, and what does / doesn't transfer

- The model is tiny and the corpus is short and repetitive; the low loss is
  **not** a quality signal. Nothing about the *mechanism* depends on scale: the
  same capture/verify machinery applies to a real run. What changes at scale is
  the cost of re-derivation (re-running a real speedrun to verify is expensive)
  and the likelihood of hitting real GPU nondeterminism (§3), which pushes the
  honest verdict from VERIFIED toward UNKNOWN.
- A production version would verify against **checkpoints** (hash of weights at
  step k) and re-derive only *segments* of the trajectory, rather than re-running
  end to end — trading full re-derivation for spot-checks, an explicitly weaker
  guarantee that should be labelled as such.

## 6. Detection coverage — what the negative controls do and do not prove

The 12-case suite proves the verifier catches **the tampers it tests** (naive
edits, re-sealed forgeries of result/loss/data, code and env mismatches). It does
**not** prove the verifier catches *all* possible tampers. Adversarial probing
beyond the suite found one real blind spot — the `meta` block (timestamp, tool)
originally sat **outside** the seal, so an edit went undetected as VERIFIED. It is
now sealed into the chain and guarded by a regression control. Sealing makes it
tamper-*evident*, not *true*: the verifier guarantees the timestamp was not
altered, never that it is honest.

One deliberate non-detection: adding an **inert** field to `config` that
`run_training` ignores stays VERIFIED after re-sealing. This is correct, not a
gap — the verifier certifies "these inputs produce this result", and an inert
change does not change the result. Only fields that actually affect the
computation are load-bearing, and altering any of those makes re-derivation
diverge (caught). The claim is reconstructibility of the result, not textual
immutability of every field.

## 7. Trusted base (TCB)

Not verified, assumed correct: Python, NumPy, Apple Accelerate/BLAS, the OS, the
CPU, and — for the verifier — its own copy of the provenance library and the
`run_training` code it executes. SHA-256 is assumed collision-resistant. The
verifier is trusted to run in an environment that faithfully reproduces the
recorded numeric environment; §3 is exactly where that assumption is weakest.

## Summary

**Achieved (this artifact):** tamper-evidence and provenance are real and
independently checkable; on a matching environment the result re-derives
bit-for-bit, and every tampering in the negative-control suite is caught.

**Honest stop:** full re-derivation across *different* hardware is UNKNOWN and
untested; the record establishes integrity and reconstructibility, not
correctness, not quality, and not accountability. That boundary is the
contribution.
