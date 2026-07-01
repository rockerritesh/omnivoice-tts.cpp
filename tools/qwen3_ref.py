#!/usr/bin/env python3
"""Numpy reference forward pass for the OmniVoice Qwen3 backbone.

Golden oracle for validating the C++/ggml implementation (P2). Loads the
`llm.*` tensors from the generator safetensors and runs a Qwen3 forward on a
fixed integer token sequence, dumping input ids + final hidden states so the
C++ engine can be compared bit-approximately.

Qwen3 specifics implemented: RMSNorm, per-head q/k RMSNorm (over head_dim,
pre-RoPE), decoupled head_dim (128) with GQA (16 q / 8 kv heads), NeoX-style
RoPE (theta 1e6), SwiGLU MLP. Attention mask is selectable: `full`
(bidirectional — what OmniVoice uses) or `causal` (for cross-checks).

Run:
  uv run --with safetensors --with numpy python tools/qwen3_ref.py \
    --src <snapshot> --mask full --out golden/qwen3_ref
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def find_default_snapshot() -> Path | None:
    base = Path.home() / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots"
    s = sorted(base.glob("*")) if base.exists() else []
    return s[-1] if s else None


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    # x: [..., d], w: [d]
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w


def rope_cos_sin(T: int, head_dim: int, theta: float):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(T, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)            # [T, half]
    emb = np.concatenate([freqs, freqs], -1)   # [T, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [T, H, D]; cos/sin: [T, D] -> broadcast over heads
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = np.concatenate([-x2, x1], axis=-1)   # rotate_half (NeoX)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def load_llm(src: Path):
    W = {}
    with safe_open(str(src / "model.safetensors"), framework="numpy") as f:
        for k in f.keys():
            if k.startswith("llm.") or k in ("audio_embeddings.weight", "audio_heads.weight"):
                W[k] = f.get_tensor(k).astype(np.float32)
    return W


def forward(W, cfg, input_ids: np.ndarray, mask_kind: str):
    llm = cfg["llm_config"]
    H = llm["hidden_size"]; nL = llm["num_hidden_layers"]
    nH = llm["num_attention_heads"]; nKV = llm["num_key_value_heads"]
    hd = llm["head_dim"]; eps = llm["rms_norm_eps"]
    theta = llm["rope_parameters"]["rope_theta"]
    T = input_ids.shape[0]
    rep = nH // nKV

    h = W["llm.embed_tokens.weight"][input_ids]        # [T, H]
    cos, sin = rope_cos_sin(T, hd, theta)

    if mask_kind == "causal":
        bias = np.triu(np.full((T, T), -1e30, np.float32), k=1)
    else:  # full / bidirectional
        bias = np.zeros((T, T), np.float32)

    for i in range(nL):
        p = f"llm.layers.{i}."
        r = h
        x = rmsnorm(h, W[p + "input_layernorm.weight"], eps)
        q = (x @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nH, hd)
        k = (x @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nKV, hd)
        v = (x @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nKV, hd)
        q = rmsnorm(q, W[p + "self_attn.q_norm.weight"], eps)
        k = rmsnorm(k, W[p + "self_attn.k_norm.weight"], eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = np.repeat(k, rep, axis=1)                   # GQA expand -> [T,nH,hd]
        v = np.repeat(v, rep, axis=1)
        # attention per head
        qh = q.transpose(1, 0, 2)                        # [nH,T,hd]
        kh = k.transpose(1, 0, 2)
        vh = v.transpose(1, 0, 2)
        scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(hd)  # [nH,T,T]
        scores = scores + bias[None]
        out = softmax(scores) @ vh                       # [nH,T,hd]
        out = out.transpose(1, 0, 2).reshape(T, nH * hd)  # [T, 2048]
        h = r + out @ W[p + "self_attn.o_proj.weight"].T
        r = h
        x = rmsnorm(h, W[p + "post_attention_layernorm.weight"], eps)
        gate = x @ W[p + "mlp.gate_proj.weight"].T
        up = x @ W[p + "mlp.up_proj.weight"].T
        act = (gate / (1.0 + np.exp(-gate))) * up        # SiLU * up
        h = r + act @ W[p + "mlp.down_proj.weight"].T

    h = rmsnorm(h, W["llm.norm.weight"], eps)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=find_default_snapshot())
    ap.add_argument("--mask", choices=["full", "causal"], default="full")
    ap.add_argument("--out", type=Path, default=Path("golden/qwen3_ref"))
    ap.add_argument("--tokens", type=str, default="9707,11,1879,0,151645,198,40,1079",
                    help="comma-separated input token ids")
    args = ap.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    W = load_llm(args.src)
    ids = np.array([int(t) for t in args.tokens.split(",")], dtype=np.int64)
    h = forward(W, cfg, ids, args.mask)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out) + f"_ids.npy", ids)
    np.save(str(args.out) + f"_hidden_{args.mask}.npy", h.astype(np.float32))
    print(f"mask={args.mask} T={len(ids)} hidden={h.shape}")
    print(f"hidden[0,:6]  = {h[0,:6]}")
    print(f"hidden[-1,:6] = {h[-1,:6]}")
    print(f"mean={h.mean():.6f} std={h.std():.6f} absmax={np.abs(h).max():.6f}")
    print(f"saved {args.out}_hidden_{args.mask}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
