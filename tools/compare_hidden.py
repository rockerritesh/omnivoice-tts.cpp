#!/usr/bin/env python3
"""Compare the C++ ggml hidden-state dump against the numpy golden oracle.

Both are laid out row-major [T, n_embd] as float32.
  uv run --with numpy python tools/compare_hidden.py <golden.npy> <cpp.bin>
"""
import sys
import numpy as np

golden = np.load(sys.argv[1]).astype(np.float32)          # [T, n_embd]
cpp = np.fromfile(sys.argv[2], dtype=np.float32).reshape(golden.shape)

diff = np.abs(cpp - golden)
denom = np.maximum(np.abs(golden), 1e-3)
rel = diff / denom
print(f"shape        = {golden.shape}")
print(f"golden mean/std = {golden.mean():.5f} / {golden.std():.5f}")
print(f"cpp    mean/std = {cpp.mean():.5f} / {cpp.std():.5f}")
print(f"max abs diff = {diff.max():.6f}   (at {np.unravel_index(diff.argmax(), diff.shape)})")
print(f"mean abs diff= {diff.mean():.6f}")
print(f"max rel diff = {rel.max():.6f}")
# cosine similarity per row
cos = (cpp * golden).sum(-1) / (np.linalg.norm(cpp, axis=-1) * np.linalg.norm(golden, axis=-1) + 1e-9)
print(f"row cosine   = min {cos.min():.6f}  mean {cos.mean():.6f}")

# f16 weights => expect small but nonzero error. Pass if cosine ~1 and max abs diff modest.
ok = cos.min() > 0.999 and rel.mean() < 0.05
print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
sys.exit(0 if ok else 1)
