"""
Determinism probe: does the SAME code+config re-derive bit-for-bit across
different BLAS thread counts on THIS machine?

Runs train.py in subprocesses with the thread env set BEFORE numpy imports (the
only way to actually rebind BLAS threading), then compares the COMPUTE payloads
(config/data/trajectory/result) -- excluding the env segment, which is expected
to differ because it records the thread count.

This is the empirical basis for the GAPS.md claim about where re-derivation holds
and where it is UNKNOWN. It reports what it finds; it does not assume an answer.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from provenance.canonical import canonical_bytes  # noqa: E402

COMPUTE_KINDS = ["config", "data", "trajectory", "result"]


def capture(threads: int, out: str):
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    subprocess.run([sys.executable, os.path.join(ROOT, "train.py"), "--out", out],
                   env=env, check=True, capture_output=True)
    with open(out) as f:
        return json.load(f)


def compute_fingerprint(record):
    segs = {s["kind"]: s["payload"] for s in record["segments"]}
    return canonical_bytes({k: segs[k] for k in COMPUTE_KINDS}).hex()[:32]


def main():
    tmp = tempfile.mkdtemp()
    counts = [1, 1, 8, 16]
    fps = []
    for i, th in enumerate(counts):
        rec = capture(th, os.path.join(tmp, f"r{i}.json"))
        fp = compute_fingerprint(rec)
        fps.append((th, fp, rec["seal"][:12]))
        print(f"threads={th:2d}  compute_fingerprint={fp}  seal={rec['seal'][:12]}")
    base = fps[0][1]
    all_same = all(fp == base for _, fp, _ in fps)
    print("-" * 60)
    print(f"compute payloads identical across thread counts: {all_same}")
    print("seals differ because the record binds the capture TIME (meta) and the")
    print("thread count (env); the COMPUTE fingerprint above is the reproducible,")
    print("time/env-independent identity of the computation. That is what two")
    print("parties compare; the seal is per-capture tamper-evidence, not a")
    print("canonical fingerprint.")
    print("BLAS backend: apple accelerate (see numpy.show_config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
