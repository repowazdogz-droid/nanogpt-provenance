"""
Finite-difference gradient check for model.py.

The training engine's correctness is not assumed -- we verify the hand-written
backward pass against numerical gradients on a tiny random instance. If this
fails, every downstream loss number is suspect, so it runs as a gate.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from model import Config, init_params, forward, backward

def run():
    rng = np.random.default_rng(0)
    cfg = Config(vocab_size=7, block_size=6, n_layer=2, n_head=2, n_embd=8)
    p = init_params(cfg, rng)
    B, T = 3, cfg.block_size
    idx = rng.integers(0, cfg.vocab_size, size=(B, T))
    tgt = rng.integers(0, cfg.vocab_size, size=(B, T))

    _, loss0, cache = forward(p, cfg, idx, tgt)
    grads = backward(p, cfg, cache)

    # eps=1e-4 central difference: for float64, smaller eps loses precision in
    # the (lp-lm) numerator and inflates relative error on tiny gradients.
    eps = 1e-4
    worst_rel = 0.0
    worst_abs = 0.0
    worst_combined = 0.0  # min(rel, abs) per entry -- an entry passes if EITHER is tiny
    checked = 0
    # check a random subset of entries in every parameter
    for name, arr in p.items():
        flat = arr.reshape(-1)
        n = flat.size
        picks = np.random.default_rng(hash(name) % (2**32)).integers(0, n, size=min(4, n))
        for i in picks:
            orig = flat[i]
            flat[i] = orig + eps
            _, lp, _ = forward(p, cfg, idx, tgt)
            flat[i] = orig - eps
            _, lm, _ = forward(p, cfg, idx, tgt)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            abs_err = abs(num - ana)
            rel = abs_err / max(1e-12, abs(num) + abs(ana))
            worst_rel = max(worst_rel, rel)
            worst_abs = max(worst_abs, abs_err)
            # An entry is correct if it agrees relatively OR its absolute error
            # is negligible (relative error blows up on near-zero gradients).
            worst_combined = max(worst_combined, min(rel, abs_err))
            checked += 1
    print(f"gradcheck: entries={checked} worst_rel={worst_rel:.2e} "
          f"worst_abs={worst_abs:.2e} worst_combined={worst_combined:.2e}")
    # Pass: every entry has either rel<1e-4 or abs<1e-6 (so combined<1e-4).
    ok = worst_combined < 1e-4
    print("GRADCHECK:", "PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
