#!/usr/bin/env python3
"""Export the OmniVoice (Qwen2) BPE tokenizer to a compact binary for the C++ engine.

Format (little-endian):
  magic "OVTK", u32 version=1
  u32 n_vocab;  then n_vocab * [u32 id, u32 blen, blen bytes]   (token string -> id)
  u32 n_merges; then n_merges * [u32 la, la bytes, u32 lb, lb bytes]  (rank = order)
  u32 n_added;  then n_added  * [u32 id, u32 blen, blen bytes]   (special tokens -> id)

Token/merge strings are the byte-level (GPT-2) encoded forms exactly as stored in
tokenizer.json, so the C++ side operates on identical byte strings.

  uv run --with tokenizers python tools/export_tokenizer.py --out models/tokenizer.bin
"""
import argparse, glob, json, struct
from pathlib import Path


def find_tok():
    g = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/*/tokenizer.json"))
    return g[0] if g else None


def w_str(buf, s: str):
    b = s.encode("utf-8")
    buf += struct.pack("<I", len(b)); buf += b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=find_tok())
    ap.add_argument("--out", type=Path, default=Path("models/tokenizer.bin"))
    a = ap.parse_args()
    tj = json.load(open(a.tokenizer))
    vocab = tj["model"]["vocab"]              # token -> id
    merges = tj["model"]["merges"]            # list of [l, r] (or "l r")
    added = tj.get("added_tokens", [])

    buf = bytearray()
    buf += b"OVTK"; buf += struct.pack("<I", 1)
    buf += struct.pack("<I", len(vocab))
    for tok, tid in vocab.items():
        buf += struct.pack("<I", tid); w_str(buf, tok)
    buf += struct.pack("<I", len(merges))
    for m in merges:
        l, r = m if isinstance(m, list) else m.split(" ", 1)
        w_str(buf, l); w_str(buf, r)
    buf += struct.pack("<I", len(added))
    for at in added:
        buf += struct.pack("<I", at["id"]); w_str(buf, at["content"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(buf)
    print(f"wrote {a.out}: {len(vocab)} vocab, {len(merges)} merges, {len(added)} added ({len(buf)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
