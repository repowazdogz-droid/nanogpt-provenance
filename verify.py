"""
Independent verifier for a nanogpt-provenance record.

    python3 verify.py runs/record.json

It answers, with evidence, ONE question in three honest states:

  VERIFIED  -- the chain is internally consistent, the recorded data reconstructs
               to the recorded hashes, the recorded code matches this checkout,
               and RE-RUNNING training from the recorded config re-derives the
               recorded trajectory and final val loss BIT-FOR-BIT, in an
               environment whose determinism-critical fingerprint matches.
  TAMPERED  -- the chain is broken, the data hash doesn't match the actual data,
               or the recorded numbers do NOT re-derive even though the code and
               numeric environment match the record (internally inconsistent).
  UNKNOWN   -- the verifier CANNOT settle the question: it can't reconstruct an
               input, the recorded code differs from this checkout (so a re-run
               tests different code), or a re-derivation mismatch occurs under a
               DIFFERENT numeric environment than the record declares, so the
               mismatch can't be attributed to tampering vs nondeterminism.
               UNKNOWN is never silently upgraded to VERIFIED.

The verifier RE-DERIVES by importing and running the same `run_training`
function on the recorded config. That is the external reference of the
`a-record-cannot-witness-itself` ladder: it does not trust the record's claimed
output, it recomputes it. What it establishes is reconstructibility of the
RESULT from the INPUTS -- NOT that the code is correct or the result is good.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import data as data_mod
from provenance.canonical import canonical_bytes, sha256_raw, sha256_file
from provenance.chain import recompute_chain
from provenance.capture import capture_env, SOURCE_FILES, COMPUTE_SOURCE, PROVENANCE_SOURCE
from train import run_training

ROOT = os.path.dirname(os.path.abspath(__file__))

VERIFIED, TAMPERED, UNKNOWN = "VERIFIED", "TAMPERED", "UNKNOWN"


def _segmap(record):
    return {s["kind"]: s["payload"] for s in record["segments"]}


def check_chain(record):
    ok, findings = recompute_chain(record)
    broken = [f for f in findings if f.get("index") != "seal"
              and (not f.get("prev_link_ok") or not f.get("hash_ok"))]
    seal = [f for f in findings if f.get("index") == "seal"][0]
    return {
        "name": "chain_integrity",
        "ok": ok,
        "broken_segments": [{"index": f["index"], "kind": f["kind"],
                             "prev_link_ok": f["prev_link_ok"], "hash_ok": f["hash_ok"]}
                            for f in broken],
        "seal_ok": seal["seal_ok"],
        "detail": "recomputed every segment hash and the seal from payloads",
    }


def check_data(record):
    seg = _segmap(record).get("data", {})
    # rebuild with the val_frac recorded in config, so the split matches
    cfg = _segmap(record)["config"]["config"]
    ds = data_mod.build_dataset(val_frac=cfg["val_frac"])
    recomputed = {
        "train_sha256": sha256_raw(ds.train.tobytes()),
        "val_sha256": sha256_raw(ds.val.tobytes()),
        "source_sha256": sha256_raw(ds.source_bytes),
        "vocab_size": ds.vocab_size,
        "n_train_tokens": int(len(ds.train)),
        "n_val_tokens": int(len(ds.val)),
    }
    fields = ["train_sha256", "val_sha256", "source_sha256",
              "vocab_size", "n_train_tokens", "n_val_tokens"]
    mism = {k: {"recorded": seg.get(k), "recomputed": recomputed[k]}
            for k in fields if seg.get(k) != recomputed[k]}
    return {"name": "data_reconstruction", "ok": not mism,
            "mismatches": mism,
            "detail": "rebuilt dataset from embedded corpus and recomputed hashes"}


def check_code(record):
    seg = _segmap(record).get("code", {})
    recorded = seg.get("source_sha256", {}) or {}
    compute_diffs, prov_diffs = {}, {}
    for rel in SOURCE_FILES:
        p = os.path.join(ROOT, rel)
        cur = sha256_file(p) if os.path.exists(p) else None
        if recorded.get(rel) != cur:
            bucket = compute_diffs if rel in COMPUTE_SOURCE else prov_diffs
            bucket[rel] = {"recorded": recorded.get(rel), "current": cur}
    # Only COMPUTE_SOURCE mismatches block re-derivation attribution. A verifier
    # is expected to run its own provenance library, so prov_diffs are noted but
    # not disqualifying.
    return {"name": "code_integrity", "ok": not compute_diffs,
            "source_mismatches": compute_diffs,
            "provenance_lib_differences": prov_diffs,
            "git_commit_recorded": seg.get("git_commit"),
            "git_dirty_recorded": seg.get("git_dirty"),
            "detail": "recomputed COMPUTE_SOURCE (model/train/data) hashes vs record; "
                      "provenance-library differences reported but not disqualifying"}


def check_env(record):
    seg = _segmap(record).get("env", {})
    rec_crit = seg.get("determinism_critical", {})
    cur_crit = capture_env()["determinism_critical"]
    deltas = {}
    keys = set(rec_crit) | set(cur_crit)
    for k in keys:
        if rec_crit.get(k) != cur_crit.get(k):
            deltas[k] = {"recorded": rec_crit.get(k), "current": cur_crit.get(k)}
    return {"name": "environment_fingerprint", "matches": not deltas,
            "deltas": deltas,
            "detail": "compared determinism-critical fields (threads, dtype, "
                      "numpy, platform, BLAS) to the current environment"}


def check_rederivation(record):
    cfg = _segmap(record)["config"]["config"]
    fresh = run_training(cfg)
    rec = _segmap(record)
    rec_traj = {"trajectory": rec["trajectory"]["trajectory"],
                "evals": rec["trajectory"]["evals"]}
    new_traj = {"trajectory": fresh["trajectory"], "evals": fresh["evals"]}
    traj_match = canonical_bytes(rec_traj) == canonical_bytes(new_traj)

    rec_res = {k: rec["result"][k] for k in ("final_val_loss", "final_train_loss", "n_steps")}
    new_res = {"final_val_loss": fresh["final_val_loss"],
               "final_train_loss": fresh["final_train_loss"],
               "n_steps": fresh["n_steps"]}
    res_match = canonical_bytes(rec_res) == canonical_bytes(new_res)

    first_diff = None
    if not traj_match:
        rt, nt = rec_traj["trajectory"], new_traj["trajectory"]
        for i in range(min(len(rt), len(nt))):
            if canonical_bytes(rt[i]) != canonical_bytes(nt[i]):
                first_diff = {"step": rt[i].get("step"),
                              "recorded_train_loss": rt[i]["train_loss"],
                              "recomputed_train_loss": nt[i]["train_loss"]}
                break
        if first_diff is None and len(rt) != len(nt):
            first_diff = {"length_mismatch": [len(rt), len(nt)]}

    # The reproducible, time/env-independent identity of the computation. Two
    # parties who ran the same thing get the same fingerprint even if their seals
    # differ (different capture time / environment).
    compute_fp = __import__("hashlib").sha256(canonical_bytes({
        "config": rec["config"], "data": rec["data"],
        "trajectory": rec["trajectory"], "result": rec["result"]})).hexdigest()

    return {"name": "re_derivation",
            "compute_fingerprint": compute_fp,
            "trajectory_match": traj_match,
            "result_match": res_match,
            "match": traj_match and res_match,
            "first_divergence": first_diff,
            "recorded_final_val_loss": rec_res["final_val_loss"],
            "recomputed_final_val_loss": new_res["final_val_loss"],
            "detail": "re-ran run_training(config) and compared trajectory + "
                      "final metrics bit-for-bit against the record"}


def verify_record(record, rederive=True):
    chain = check_chain(record)
    data = check_data(record)
    code = check_code(record)
    env = check_env(record)

    checks = {"chain_integrity": chain, "data_reconstruction": data,
              "code_integrity": code, "environment_fingerprint": env}

    # --- integrity gate: broken chain or wrong data hash = TAMPERED, no re-run.
    if not chain["ok"]:
        return _verdict(TAMPERED, checks,
                        "hash chain does not recompute: a segment or the seal was altered")
    if not data["ok"]:
        return _verdict(TAMPERED, checks,
                        "recorded data hash does not match the actual reconstructed data")

    # --- can we even re-derive faithfully?
    if not code["ok"]:
        return _verdict(UNKNOWN, checks,
                        "recorded code differs from this checkout; a re-run would test "
                        "different code, so the recorded code->result link is unconfirmable")

    if not rederive:
        return _verdict(UNKNOWN, checks, "re-derivation skipped (--no-rederive)")

    red = check_rederivation(record)
    checks["re_derivation"] = red

    if red["match"]:
        if env["matches"]:
            return _verdict(VERIFIED, checks,
                            "chain intact, data reconstructs, code matches, and the "
                            "result re-derives bit-for-bit in a matching environment")
        # Numbers reproduced exactly, but the environment fingerprint differed.
        # The claim re-derived here, so this is VERIFIED; we still surface that
        # reproduction in the record's DECLARED environment was not tested.
        return _verdict(VERIFIED, checks,
                        "result re-derives bit-for-bit, but the numeric-environment "
                        "fingerprint differs from the record (see env deltas); "
                        "reproduction in the record's declared environment is UNKNOWN")

    # re-derivation mismatch
    if env["matches"]:
        return _verdict(TAMPERED, checks,
                        "result does NOT re-derive even though code and numeric "
                        "environment match the record: the record is internally inconsistent")
    return _verdict(UNKNOWN, checks,
                    "result does not re-derive AND the numeric environment differs "
                    "from the record; the mismatch cannot be attributed to tampering "
                    "vs environment nondeterminism")


def _verdict(v, checks, reason):
    return {"verdict": v, "reason": reason, "checks": checks}


def print_report(report, record):
    v = report["verdict"]
    bar = {"VERIFIED": "\033[92m", "TAMPERED": "\033[91m", "UNKNOWN": "\033[93m"}.get(v, "")
    rst = "\033[0m"
    print(f"\nseal: {record.get('seal')}")
    print("-" * 72)
    c = report["checks"]
    ci = c["chain_integrity"]
    print(f"[{'PASS' if ci['ok'] else 'FAIL'}] chain_integrity        "
          f"seal_ok={ci['seal_ok']} broken={len(ci['broken_segments'])}")
    if ci["broken_segments"]:
        for b in ci["broken_segments"]:
            print(f"        -> segment[{b['index']}] kind={b['kind']} "
                  f"prev_link_ok={b['prev_link_ok']} hash_ok={b['hash_ok']}")
    dr = c["data_reconstruction"]
    print(f"[{'PASS' if dr['ok'] else 'FAIL'}] data_reconstruction    "
          f"mismatches={list(dr['mismatches'].keys()) or 'none'}")
    cc = c["code_integrity"]
    print(f"[{'PASS' if cc['ok'] else 'WARN'}] code_integrity         "
          f"source_mismatches={list(cc['source_mismatches'].keys()) or 'none'}")
    ev = c["environment_fingerprint"]
    print(f"[{'PASS' if ev['matches'] else 'WARN'}] environment_fingerprint "
          f"deltas={list(ev['deltas'].keys()) or 'none'}")
    if "re_derivation" in c:
        rd = c["re_derivation"]
        print(f"[{'PASS' if rd['match'] else 'FAIL'}] re_derivation          "
              f"trajectory_match={rd['trajectory_match']} result_match={rd['result_match']}")
        print(f"        compute_fingerprint = {rd['compute_fingerprint'][:32]}  "
              f"(reproducible identity; seal differs per capture)")
        print(f"        recorded final_val_loss   = {rd['recorded_final_val_loss']}")
        print(f"        recomputed final_val_loss = {rd['recomputed_final_val_loss']}")
        if rd["first_divergence"]:
            print(f"        first_divergence: {rd['first_divergence']}")
    print("-" * 72)
    print(f"VERDICT: {bar}{v}{rst}")
    print(f"  {report['reason']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--no-rederive", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = ap.parse_args()
    with open(args.record) as f:
        record = json.load(f)
    report = verify_record(record, rederive=not args.no_rederive)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report, record)
    sys.exit({"VERIFIED": 0, "UNKNOWN": 2, "TAMPERED": 1}[report["verdict"]])


if __name__ == "__main__":
    main()
