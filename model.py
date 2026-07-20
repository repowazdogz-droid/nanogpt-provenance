"""
A tiny, faithful GPT in pure NumPy -- the "same machinery" as nanoGPT at a scale
that runs on a laptop CPU in seconds:

  token embedding + learned positional embedding
  N pre-LayerNorm transformer blocks (causal multi-head self-attention + GELU MLP)
  final LayerNorm + linear LM head
  cross-entropy loss, AdamW, warmup+cosine LR schedule, seeded init & batching

Everything is float64 and single-threaded-friendly so a run is deterministic and
a verifier can re-derive it bit-for-bit (see SCOPE.md). Forward and backward are
written out by hand; correctness is checked by a finite-difference gradient test
in tests/gradcheck.py -- the training engine itself is verified, not assumed.

This is a SIMPLIFIED transformer (untied LM head, GELU-tanh, no dropout at the
scale used). It is not modded-nanoGPT; it exercises the same determinants of a
result (data, config, seed, code, environment) which is what the provenance layer
is about.
"""
from __future__ import annotations

import numpy as np

DT = np.float64


# ---------------------------------------------------------------- primitives
def gelu(x):
    # GPT-2 tanh approximation
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    x3 = x**3
    inner = c * (x + 0.044715 * x3)
    t = np.tanh(inner)
    dinner = c * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc**2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    out = xhat * g + b
    cache = (xhat, inv, g)
    return out, cache


def layernorm_grad(dout, cache):
    xhat, inv, g = cache
    C = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, C).sum(0)
    db = dout.reshape(-1, C).sum(0)
    dxhat = dout * g
    dx = inv / C * (C * dxhat
                    - dxhat.sum(axis=-1, keepdims=True)
                    - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True))
    return dx, dg, db


def softmax_lastdim(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- model
class Config:
    def __init__(self, vocab_size, block_size, n_layer=2, n_head=2, n_embd=64):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

    def as_dict(self):
        return dict(vocab_size=self.vocab_size, block_size=self.block_size,
                    n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd)


def init_params(cfg: Config, rng: np.random.Generator) -> dict:
    C = cfg.n_embd
    p = {}
    p["wte"] = (rng.standard_normal((cfg.vocab_size, C)) * 0.02).astype(DT)
    p["wpe"] = (rng.standard_normal((cfg.block_size, C)) * 0.02).astype(DT)
    for l in range(cfg.n_layer):
        p[f"ln1_g_{l}"] = np.ones(C, DT); p[f"ln1_b_{l}"] = np.zeros(C, DT)
        p[f"attn_qkv_w_{l}"] = (rng.standard_normal((C, 3 * C)) * 0.02).astype(DT)
        p[f"attn_qkv_b_{l}"] = np.zeros(3 * C, DT)
        p[f"attn_proj_w_{l}"] = (rng.standard_normal((C, C)) * 0.02).astype(DT)
        p[f"attn_proj_b_{l}"] = np.zeros(C, DT)
        p[f"ln2_g_{l}"] = np.ones(C, DT); p[f"ln2_b_{l}"] = np.zeros(C, DT)
        p[f"mlp_fc_w_{l}"] = (rng.standard_normal((C, 4 * C)) * 0.02).astype(DT)
        p[f"mlp_fc_b_{l}"] = np.zeros(4 * C, DT)
        p[f"mlp_proj_w_{l}"] = (rng.standard_normal((4 * C, C)) * 0.02).astype(DT)
        p[f"mlp_proj_b_{l}"] = np.zeros(C, DT)
    p["lnf_g"] = np.ones(C, DT); p["lnf_b"] = np.zeros(C, DT)
    p["lm_head_w"] = (rng.standard_normal((C, cfg.vocab_size)) * 0.02).astype(DT)
    p["lm_head_b"] = np.zeros(cfg.vocab_size, DT)
    return p


def _attention(x, p, l, cfg, cache):
    B, T, C = x.shape
    nh, hs = cfg.n_head, C // cfg.n_head
    qkv = x @ p[f"attn_qkv_w_{l}"] + p[f"attn_qkv_b_{l}"]     # (B,T,3C)
    q, k, v = np.split(qkv, 3, axis=-1)
    def heads(z): return z.reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
    q, k, v = heads(q), heads(k), heads(v)                   # (B,nh,T,hs)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hs)        # (B,nh,T,T)
    mask = np.triu(np.ones((T, T), bool), k=1)
    att = np.where(mask, -np.inf, att)
    p_att = softmax_lastdim(att)
    y = p_att @ v                                            # (B,nh,T,hs)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = y @ p[f"attn_proj_w_{l}"] + p[f"attn_proj_b_{l}"]
    cache[f"attn_{l}"] = (x, q, k, v, p_att, y, nh, hs)
    return out


def _attention_grad(dout, p, l, cfg, cache, grads):
    x, q, k, v, p_att, y, nh, hs = cache[f"attn_{l}"]
    B, T, C = x.shape
    grads[f"attn_proj_w_{l}"] = y.reshape(-1, C).T @ dout.reshape(-1, C)
    grads[f"attn_proj_b_{l}"] = dout.reshape(-1, C).sum(0)
    dy = dout @ p[f"attn_proj_w_{l}"].T                       # (B,T,C)
    dy = dy.reshape(B, T, nh, hs).transpose(0, 2, 1, 3)      # (B,nh,T,hs)
    dp_att = dy @ v.transpose(0, 1, 3, 2)                     # (B,nh,T,T)
    dv = p_att.transpose(0, 1, 3, 2) @ dy                    # (B,nh,T,hs)
    # softmax backward (per row)
    ds = p_att * (dp_att - (dp_att * p_att).sum(axis=-1, keepdims=True))
    ds = ds / np.sqrt(hs)
    dq = ds @ k                                              # (B,nh,T,hs)
    dk = ds.transpose(0, 1, 3, 2) @ q                        # (B,nh,T,hs)
    def unheads(z): return z.transpose(0, 2, 1, 3).reshape(B, T, C)
    dqkv = np.concatenate([unheads(dq), unheads(dk), unheads(dv)], axis=-1)  # (B,T,3C)
    grads[f"attn_qkv_w_{l}"] = x.reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
    grads[f"attn_qkv_b_{l}"] = dqkv.reshape(-1, 3 * C).sum(0)
    dx = dqkv @ p[f"attn_qkv_w_{l}"].T
    return dx


def forward(p, cfg, idx, targets=None):
    B, T = idx.shape
    C = cfg.n_embd
    cache = {}
    x = p["wte"][idx] + p["wpe"][:T]                          # (B,T,C)
    cache["idx"] = idx
    for l in range(cfg.n_layer):
        h, c1 = layernorm(x, p[f"ln1_g_{l}"], p[f"ln1_b_{l}"])
        cache[f"ln1_{l}"] = c1
        x = x + _attention(h, p, l, cfg, cache)
        h, c2 = layernorm(x, p[f"ln2_g_{l}"], p[f"ln2_b_{l}"])
        cache[f"ln2_{l}"] = c2
        fc = h @ p[f"mlp_fc_w_{l}"] + p[f"mlp_fc_b_{l}"]      # (B,T,4C)
        act = gelu(fc)
        mlp = act @ p[f"mlp_proj_w_{l}"] + p[f"mlp_proj_b_{l}"]
        cache[f"mlp_{l}"] = (h, fc, act)
        x = x + mlp
    xf, cf = layernorm(x, p["lnf_g"], p["lnf_b"])
    cache["lnf"] = cf
    logits = xf @ p["lm_head_w"] + p["lm_head_b"]            # (B,T,V)
    cache["xf"] = xf
    if targets is None:
        return logits, None, cache
    V = cfg.vocab_size
    flat = logits.reshape(-1, V)
    probs = softmax_lastdim(flat)
    tgt = targets.reshape(-1)
    n = tgt.shape[0]
    loss = -np.log(probs[np.arange(n), tgt] + 1e-12).mean()
    cache["softmax"] = (probs, tgt, n, logits.shape)
    return logits, float(loss), cache


def backward(p, cfg, cache):
    grads = {}
    probs, tgt, n, lshape = cache["softmax"]
    B, T, V = lshape
    dlogits = probs.copy()
    dlogits[np.arange(n), tgt] -= 1.0
    dlogits /= n
    dlogits = dlogits.reshape(B, T, V)
    xf = cache["xf"]; C = cfg.n_embd
    grads["lm_head_w"] = xf.reshape(-1, C).T @ dlogits.reshape(-1, V)
    grads["lm_head_b"] = dlogits.reshape(-1, V).sum(0)
    dxf = dlogits @ p["lm_head_w"].T
    dx, dg, db = layernorm_grad(dxf, cache["lnf"])
    grads["lnf_g"], grads["lnf_b"] = dg, db
    for l in reversed(range(cfg.n_layer)):
        h, fc, act = cache[f"mlp_{l}"]
        dmlp = dx
        grads[f"mlp_proj_w_{l}"] = act.reshape(-1, 4 * C).T @ dmlp.reshape(-1, C)
        grads[f"mlp_proj_b_{l}"] = dmlp.reshape(-1, C).sum(0)
        dact = dmlp @ p[f"mlp_proj_w_{l}"].T
        dfc = dact * gelu_grad(fc)
        grads[f"mlp_fc_w_{l}"] = h.reshape(-1, C).T @ dfc.reshape(-1, 4 * C)
        grads[f"mlp_fc_b_{l}"] = dfc.reshape(-1, 4 * C).sum(0)
        dh = dfc @ p[f"mlp_fc_w_{l}"].T
        dln2, dg2, db2 = layernorm_grad(dh, cache[f"ln2_{l}"])
        grads[f"ln2_g_{l}"], grads[f"ln2_b_{l}"] = dg2, db2
        dx = dx + dln2                       # residual
        dattn = dx
        dh_attn = _attention_grad(dattn, p, l, cfg, cache, grads)
        dln1, dg1, db1 = layernorm_grad(dh_attn, cache[f"ln1_{l}"])
        grads[f"ln1_g_{l}"], grads[f"ln1_b_{l}"] = dg1, db1
        dx = dx + dln1                       # residual
    # embeddings
    idx = cache["idx"]
    grads["wpe"] = np.zeros_like(p["wpe"])
    T = idx.shape[1]
    grads["wpe"][:T] = dx.sum(axis=0)
    grads["wte"] = np.zeros_like(p["wte"])
    np.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, C))
    return grads


# ---------------------------------------------------------------- optimizer
NO_DECAY = ("_b_", "ln", "lm_head_b", "wte", "wpe")  # biases, norms, embeddings


def wants_decay(name: str) -> bool:
    return name.endswith("_w") or "_w_" in name


class AdamW:
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            update = mhat / (np.sqrt(vhat) + self.eps)
            if wants_decay(k):
                update = update + self.wd * params[k]
            params[k] = params[k] - lr * update


def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    ratio = (step - warmup) / max(1, (total - warmup))
    coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return lr_min + coeff * (lr_max - lr_min)
