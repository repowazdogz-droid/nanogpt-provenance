"""
Negative controls: deliberately tamper a VERIFIED record and prove the verifier
catches each one with the RIGHT verdict. If any tampering slips through as
VERIFIED, that is a FINDING, not a pass -- the script fails.

Two adversary strengths:
  NAIVE   -- edit a payload, leave the stored hashes as-is (chain breaks).
  RESEAL  -- edit a payload AND recompute all hashes + seal so the chain is
             internally consistent again (a forgery). These must be caught by
             the SEMANTIC checks: data-hash reconstruction and re-derivation.

Also exercises the honest UNKNOWN states.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provenance.chain import reseal
from verify import verify_record, VERIFIED, TAMPERED, UNKNOWN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "runs", "record.json")


def load():
    with open(RECORD) as f:
        return json.load(f)


def segset(record, kind, key, value):
    r = copy.deepcopy(record)
    for s in r["segments"]:
        if s["kind"] == kind:
            s["payload"][key] = value
    return r


def result():
    return {"pass": 0, "fail": 0, "lines": []}


def expect(res, name, report, want_verdict, want_check_fail=None):
    v = report["verdict"]
    ok = (v == want_verdict)
    detail = ""
    if want_check_fail:
        chk = report["checks"].get(want_check_fail, {})
        # a check "failed" if ok is False, matches is False, or match is False
        failed = (chk.get("ok") is False or chk.get("matches") is False
                  or chk.get("match") is False)
        ok = ok and failed
        detail = f" via {want_check_fail}={'caught' if failed else 'NOT caught'}"
    res["pass" if ok else "fail"] += 1
    mark = "PASS" if ok else "FAIL"
    res["lines"].append(f"[{mark}] {name:38s} -> {v}{detail}")


def main():
    if not os.path.exists(RECORD):
        print("run train.py first to produce runs/record.json"); return 1
    good = load()
    res = result()

    # 0) sanity: the clean record verifies
    expect(res, "clean record", verify_record(good), VERIFIED)

    # --- NAIVE tampers: chain must break ---
    # (a) change a logged loss mid-trajectory
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "trajectory":
            s["payload"]["trajectory"][100]["train_loss"] = 0.0
    expect(res, "naive: edit logged loss (step100)", verify_record(t),
           TAMPERED, "chain_integrity")

    # (b) change a config value (learning rate)
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "config":
            s["payload"]["config"]["lr_max"] = 0.5
    expect(res, "naive: swap config lr_max", verify_record(t),
           TAMPERED, "chain_integrity")

    # (c) change a data hash
    t = segset(good, "data", "train_sha256", "deadbeef" * 8)
    expect(res, "naive: alter data train_sha256", verify_record(t),
           TAMPERED, "chain_integrity")

    # (d) change the claimed final val loss
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "result":
            s["payload"]["final_val_loss"] = 0.001
    expect(res, "naive: lower claimed final_val_loss", verify_record(t),
           TAMPERED, "chain_integrity")

    # --- RESEAL forgeries: chain is made consistent; semantics must catch ---
    # (e) lower final val loss + reseal -> re-derivation mismatch (env matches)
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "result":
            s["payload"]["final_val_loss"] = 0.001
    t = reseal(t)
    rep = verify_record(t)
    # chain now passes; caught by re_derivation
    expect(res, "RESEAL: forge final_val_loss", rep, TAMPERED, "re_derivation")

    # (f) alter data hash + reseal -> data_reconstruction mismatch
    t = reseal(segset(good, "data", "train_sha256", "deadbeef" * 8))
    expect(res, "RESEAL: forge data train_sha256", verify_record(t),
           TAMPERED, "data_reconstruction")

    # (g) forge a mid-trajectory loss + reseal -> re-derivation mismatch
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "trajectory":
            s["payload"]["trajectory"][100]["train_loss"] = 1.2345
    t = reseal(t)
    expect(res, "RESEAL: forge logged loss (step100)", verify_record(t),
           TAMPERED, "re_derivation")

    # --- UNKNOWN states (honest 'cannot settle') ---
    # (h) code differs from record + reseal -> cannot re-derive recorded code
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "code":
            # forger claims a different model.py was used (or verifier is on a
            # different checkout): source hash no longer matches this tree.
            s["payload"]["source_sha256"]["model.py"] = "0" * 64
    t = reseal(t)
    expect(res, "UNKNOWN: recorded code != checkout", verify_record(t),
           UNKNOWN, "code_integrity")

    # (i) declared foreign numeric environment + reseal, numbers still re-derive
    #     -> VERIFIED (bits matched) but env delta surfaced. This documents that
    #     a PURE env-driven UNKNOWN needs a real re-derivation mismatch (GPU),
    #     which this deterministic CPU/BLAS cannot produce (see GAPS.md).
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "env":
            s["payload"]["determinism_critical"]["platform_machine"] = "x86_64-cuda"
            s["payload"]["determinism_critical"]["blas_backend"] = "cublas"
    t = reseal(t)
    rep = verify_record(t)
    env_delta_seen = bool(rep["checks"]["environment_fingerprint"]["deltas"])
    ok = (rep["verdict"] == VERIFIED and env_delta_seen)
    res["pass" if ok else "fail"] += 1
    res["lines"].append(
        f"[{'PASS' if ok else 'FAIL'}] {'env delta surfaced, bits re-derived':38s} "
        f"-> {rep['verdict']} (env_delta={env_delta_seen})")

    # (j0) edit the sealed meta timestamp -> chain must break (regression guard
    #      for a blind spot found by adversarial probing: meta used to sit
    #      OUTSIDE the seal and an edit went undetected as VERIFIED).
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "meta":
            s["payload"]["captured_unix_time"] = 0.0
    expect(res, "naive: edit sealed meta timestamp", verify_record(t),
           TAMPERED, "chain_integrity")

    # (j) forge final_val_loss AND declare a foreign environment + reseal.
    #     Re-derivation now mismatches, but because the numeric environment does
    #     not match the record, the verifier CANNOT attribute the mismatch to
    #     tampering vs nondeterminism -> honest UNKNOWN, not TAMPERED. This is the
    #     real cross-hardware behavior: tamper-detection REQUIRES a matching env.
    t = copy.deepcopy(good)
    for s in t["segments"]:
        if s["kind"] == "result":
            s["payload"]["final_val_loss"] = 0.001
        if s["kind"] == "env":
            s["payload"]["determinism_critical"]["platform_machine"] = "x86_64-cuda"
    t = reseal(t)
    expect(res, "UNKNOWN: forged loss under foreign env", verify_record(t),
           UNKNOWN, "re_derivation")

    print("\n".join(res["lines"]))
    print("-" * 72)
    print(f"negative controls: {res['pass']} passed, {res['fail']} failed")
    return 0 if res["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
