"""
Hash-chained provenance record.

A record is an ordered list of SEGMENTS. Each segment has a `kind`, a `payload`
(any canonicalizable object), and a `prev` hash. Its own hash is:

    seg_hash = sha256({"kind":..., "payload":..., "prev": <prev seg_hash>})

so the segments form a chain: altering ANY earlier payload changes every later
seg_hash, and the final `seal` (the last seg_hash) no longer matches. This is the
tamper-evidence: you cannot edit the middle of the record and keep the seal.

The record also carries `seal` at the top level -- the author's committed final
hash. A verifier recomputes the whole chain and checks that its recomputed final
hash equals the stored `seal`. If not: TAMPERED.

Genesis segment has prev = GENESIS ("0"*64), a fixed public constant, so the
chain has a well-defined, submitter-independent root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import sha256_hex

GENESIS = "0" * 64


def seg_hash(kind: str, payload: Any, prev: str) -> str:
    return sha256_hex({"kind": kind, "payload": payload, "prev": prev})


@dataclass
class Chain:
    segments: list[dict] = field(default_factory=list)

    def append(self, kind: str, payload: Any) -> str:
        prev = self.segments[-1]["hash"] if self.segments else GENESIS
        h = seg_hash(kind, payload, prev)
        self.segments.append({"kind": kind, "payload": payload, "prev": prev, "hash": h})
        return h

    @property
    def seal(self) -> str:
        if not self.segments:
            raise ValueError("empty chain has no seal")
        return self.segments[-1]["hash"]

    def to_record(self) -> dict:
        return {
            "format": "nanogpt-provenance/v1",
            "genesis": GENESIS,
            "segments": self.segments,
            "seal": self.seal,
        }


def reseal(record: dict) -> dict:
    """Recompute EVERY segment hash + the seal so an edited payload becomes
    internally consistent again.

    This is the ADVERSARY's tool, included deliberately: a sophisticated forger
    does not just edit a value and leave a broken hash -- they re-seal. This
    function lets the negative controls prove that re-sealing defeats the naive
    chain check, and that the SEMANTIC checks (data-hash reconstruction and
    re-derivation) are what actually catch such a forgery. Returns a new record.
    """
    import copy
    r = copy.deepcopy(record)
    prev = r.get("genesis", GENESIS)
    for seg in r["segments"]:
        seg["prev"] = prev
        seg["hash"] = seg_hash(seg["kind"], seg["payload"], prev)
        prev = seg["hash"]
    r["seal"] = prev
    return r


def recompute_chain(record: dict) -> tuple[bool, list[dict]]:
    """Recompute every segment hash from its payload + the running prev.

    Returns (chain_ok, findings). chain_ok is True iff:
      * each stored segment.prev equals the previous recomputed hash
        (first segment.prev == GENESIS),
      * each stored segment.hash equals seg_hash(kind, payload, prev), and
      * record.seal equals the final recomputed hash.

    findings is a per-segment list of what matched / mismatched -- evidence, so
    the verifier can point at the exact broken link, not just say "bad".
    """
    findings: list[dict] = []
    segs = record.get("segments", [])
    genesis = record.get("genesis", GENESIS)
    prev = genesis
    chain_ok = True

    for i, seg in enumerate(segs):
        kind = seg.get("kind")
        payload = seg.get("payload")
        stored_prev = seg.get("prev")
        stored_hash = seg.get("hash")

        prev_ok = stored_prev == prev
        recomputed = seg_hash(kind, payload, prev)
        # A tampered payload is caught by comparing against the hash the AUTHOR
        # stored (stored_hash). We recompute from the running `prev` so a broken
        # earlier link also surfaces here.
        hash_ok = stored_hash == recomputed
        seg_ok = prev_ok and hash_ok
        chain_ok = chain_ok and seg_ok

        findings.append(
            {
                "index": i,
                "kind": kind,
                "prev_link_ok": prev_ok,
                "hash_ok": hash_ok,
                "stored_hash": stored_hash,
                "recomputed_hash": recomputed,
            }
        )
        # Continue from the AUTHOR's stored hash so downstream links are checked
        # against what the author actually committed (isolates the first break).
        prev = stored_hash

    seal_ok = record.get("seal") == prev
    findings.append({"index": "seal", "seal_ok": seal_ok,
                     "stored_seal": record.get("seal"), "recomputed_seal": prev})
    chain_ok = chain_ok and seal_ok
    return chain_ok, findings
