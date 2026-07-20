"""
Train the tiny GPT and emit a tamper-evident, hash-chained provenance record.

    python3 train.py                 # capture a run -> runs/record.json
    python3 train.py --out runs/x.json --seed 7

The training core (`run_training`) is a PURE function of the config: same config
-> same trajectory -> same final val loss, given the same numeric environment.
The verifier imports and calls this SAME function to re-derive the result, so the
record is checked against a real recomputation, not against itself.

`run_training` returns the full trajectory + final metrics as plain floats. The
capture layer (`capture_and_seal`) wraps them into the provenance chain.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data as data_mod
from model import Config, init_params, forward, backward, AdamW, lr_at
from provenance.canonical import sha256_raw
from provenance.capture import capture_code, capture_deps, capture_env
from provenance.chain import Chain

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    "seed": 1337,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "block_size": 32,
    "batch_size": 16,
    "max_steps": 300,
    "warmup_steps": 30,
    "lr_max": 1e-2,
    "lr_min": 1e-3,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eval_interval": 50,
    "val_frac": 0.1,
}


def _eval_val_loss(p, cfg, val, block_size):
    """Deterministic full-val loss over contiguous, non-overlapping blocks."""
    losses = []
    n = len(val)
    step = block_size
    for start in range(0, n - block_size - 1, step):
        x = val[start:start + block_size][None, :]
        y = val[start + 1:start + 1 + block_size][None, :]
        _, loss, _ = forward(p, cfg, x, y)
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def run_training(config: dict) -> dict:
    """Pure function: config -> {trajectory, evals, final_*}. Deterministic given
    the numeric environment (float64, thread config -- see capture_env)."""
    cfg_seed = int(config["seed"])
    ds = data_mod.build_dataset(val_frac=config["val_frac"])
    cfg = Config(vocab_size=ds.vocab_size, block_size=config["block_size"],
                 n_layer=config["n_layer"], n_head=config["n_head"],
                 n_embd=config["n_embd"])

    rng = np.random.default_rng(cfg_seed)
    p = init_params(cfg, rng)
    opt = AdamW(p, lr=config["lr_max"],
                betas=(config["beta1"], config["beta2"]),
                weight_decay=config["weight_decay"])

    trajectory = []  # per-step train loss
    evals = []       # periodic val loss
    for step in range(config["max_steps"]):
        x, y = data_mod.get_batch(ds.train, cfg.block_size, config["batch_size"], rng)
        _, loss, cache = forward(p, cfg, x, y)
        grads = backward(p, cfg, cache)
        lr = lr_at(step, config["warmup_steps"], config["max_steps"],
                   config["lr_max"], config["lr_min"])
        opt.step(p, grads, lr)
        trajectory.append({"step": step, "lr": float(lr), "train_loss": float(loss)})
        if step % config["eval_interval"] == 0:
            vl = _eval_val_loss(p, cfg, ds.val, cfg.block_size)
            evals.append({"step": step, "val_loss": vl})

    final_val = _eval_val_loss(p, cfg, ds.val, cfg.block_size)
    final_train = trajectory[-1]["train_loss"]
    return {
        "trajectory": trajectory,
        "evals": evals,
        "final_val_loss": final_val,
        "final_train_loss": final_train,
        "n_steps": config["max_steps"],
        "data_meta": {
            "tokenizer": "char",
            "vocab_size": ds.vocab_size,
            "n_train_tokens": int(len(ds.train)),
            "n_val_tokens": int(len(ds.val)),
            "train_sha256": sha256_raw(ds.train.tobytes()),
            "val_sha256": sha256_raw(ds.val.tobytes()),
            "source_sha256": sha256_raw(ds.source_bytes),
        },
    }


def capture_and_seal(config: dict, result: dict) -> dict:
    """Wrap a training result into a hash-chained provenance record."""
    chain = Chain()
    chain.append("config", {"config": config, "model": {
        "vocab_size": result["data_meta"]["vocab_size"],
        "n_layer": config["n_layer"], "n_head": config["n_head"],
        "n_embd": config["n_embd"], "block_size": config["block_size"]}})
    chain.append("data", result["data_meta"])
    chain.append("code", {**capture_code(ROOT), **capture_deps()})
    chain.append("env", capture_env())
    chain.append("trajectory", {"trajectory": result["trajectory"],
                                "evals": result["evals"]})
    # Contextual metadata. Sealed into the chain so it is tamper-EVIDENT (a later
    # edit breaks the seal). Note: sealing makes it unaltered, NOT true -- the
    # verifier does not and cannot check that the timestamp is honest; it only
    # guarantees it was not changed after capture. Integrity, not truth.
    chain.append("meta", {
        "captured_unix_time": time.time(),
        "tool": "nanogpt-provenance/train.py",
    })
    # The CLAIM, bound into the seal (kept last so the seal binds the result).
    chain.append("result", {
        "final_val_loss": result["final_val_loss"],
        "final_train_loss": result["final_train_loss"],
        "n_steps": result["n_steps"],
    })
    return chain.to_record()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/record.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    t0 = time.time()
    result = run_training(config)
    dt = time.time() - t0
    record = capture_and_seal(config, result)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    print(f"trained {config['max_steps']} steps in {dt:.1f}s")
    print(f"final_train_loss = {result['final_train_loss']:.6f}")
    print(f"final_val_loss   = {result['final_val_loss']:.6f}")
    print(f"seal = {record['seal']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
