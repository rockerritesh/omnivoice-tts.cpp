#!/usr/bin/env python3
"""Convert a k2-fsa/OmniVoice checkpoint to GGUF for the C++/ggml engine.

Emits two files:
  * omnivoice-generator.gguf  -- Qwen3-0.6B backbone (renamed to llama.cpp
    convention) + audio_embeddings + audio_heads + OmniVoice contracts as KV.
  * omnivoice-codec.gguf      -- Higgs Audio V2 tokenizer weights (DAC decoder,
    RVQ quantizer, and encoder/semantic paths) in native names, for a custom
    ggml loader.

See MODEL_SPEC.md for the full tensor inventory this mirrors.

Run:
  uv run --with gguf --with safetensors --with numpy \
    python tools/convert_omnivoice_to_gguf.py --src <hf-snapshot-dir> --out <dir>

The default --src auto-resolves the cached k2-fsa/OmniVoice snapshot.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_default_snapshot() -> Path | None:
    base = Path.home() / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots"
    snaps = sorted(base.glob("*")) if base.exists() else []
    return snaps[-1] if snaps else None


def cast_for_gguf(name: str, arr: np.ndarray, want_f16: bool) -> np.ndarray:
    """f16 for 2D+ weight matrices; keep 1-D vectors (norms/bias/alpha) f32."""
    if arr.dtype == np.float32 and want_f16 and arr.ndim >= 2:
        # keep RVQ codebooks and tiny tables in f32 for numerical safety
        if name.endswith("codebook.embed") or arr.size < 4096:
            return arr
        return arr.astype(np.float16)
    return arr


def iter_tensors(st_path: Path):
    with safe_open(str(st_path), framework="numpy") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


# ---------------------------------------------------------------------------
# generator (Qwen3 + audio io)
# ---------------------------------------------------------------------------
def qwen3_name(key: str) -> str | None:
    """Map an OmniVoice `llm.*` tensor name to llama.cpp GGUF convention.

    Returns None for tensors that should be dropped (none here) and raises for
    unexpected names so the checkpoint layout is validated, not silently skipped.
    """
    if key == "llm.embed_tokens.weight":
        return "token_embd.weight"
    if key == "llm.norm.weight":
        return "output_norm.weight"
    if key.startswith("llm.layers."):
        _, _, rest = key.partition("llm.layers.")
        idx, _, tail = rest.partition(".")
        table = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if tail not in table:
            raise KeyError(f"unexpected qwen3 tensor: {key}")
        return f"blk.{idx}.{table[tail]}"
    raise KeyError(f"unexpected generator tensor: {key}")


def convert_generator(src: Path, out: Path, want_f16: bool) -> None:
    cfg = json.loads((src / "config.json").read_text())
    llm = cfg["llm_config"]
    st = src / "model.safetensors"

    writer = gguf.GGUFWriter(str(out), arch="qwen3")
    writer.add_name("OmniVoice-Generator")

    # --- Qwen3 hparams (llama.cpp-compatible) ---
    writer.add_context_length(llm["max_position_embeddings"])
    writer.add_embedding_length(llm["hidden_size"])
    writer.add_block_count(llm["num_hidden_layers"])
    writer.add_feed_forward_length(llm["intermediate_size"])
    writer.add_head_count(llm["num_attention_heads"])
    writer.add_head_count_kv(llm["num_key_value_heads"])
    writer.add_key_length(llm["head_dim"])
    writer.add_value_length(llm["head_dim"])
    writer.add_rope_freq_base(llm["rope_parameters"]["rope_theta"])
    writer.add_layer_norm_rms_eps(llm["rms_norm_eps"])
    writer.add_vocab_size(llm["vocab_size"])

    # --- OmniVoice-specific contracts (custom KV namespace) ---
    n_cb = cfg["num_audio_codebook"]
    vocab = cfg["audio_vocab_size"]
    writer.add_uint32("omnivoice.num_audio_codebooks", n_cb)
    writer.add_uint32("omnivoice.audio_vocab_size", vocab)
    writer.add_uint32("omnivoice.audio_mask_id", cfg["audio_mask_id"])
    writer.add_uint32("omnivoice.sample_rate", 24000)
    writer.add_uint32("omnivoice.hop_length", 960)
    writer.add_uint32("omnivoice.frame_rate", 25)
    writer.add_bool("omnivoice.non_causal_attention", True)
    writer.add_array("omnivoice.codebook_layer_offsets",
                     [c * vocab for c in range(n_cb)])
    writer.add_array("omnivoice.audio_codebook_weights",
                     [int(w) for w in cfg["audio_codebook_weights"]])

    # --- tensors ---
    n = 0
    for key, arr in iter_tensors(st):
        if key == "codebook_layer_offsets":
            continue  # stored as KV metadata instead (I64 tensors unsupported)
        if key in ("audio_embeddings.weight", "audio_heads.weight"):
            name = key  # keep native; custom C++ loader reads these
        else:
            name = qwen3_name(key)
        writer.add_tensor(name, cast_for_gguf(key, arr, want_f16))
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  wrote {out.name}: {n} tensors")


# ---------------------------------------------------------------------------
# codec (Higgs Audio V2 tokenizer)
# ---------------------------------------------------------------------------
def convert_codec(src: Path, out: Path, want_f16: bool, conv_f32: bool = False) -> None:
    cfg = json.loads((src / "audio_tokenizer/config.json").read_text())
    ac = cfg["acoustic_model_config"]
    st = src / "audio_tokenizer/model.safetensors"

    writer = gguf.GGUFWriter(str(out), arch="omnivoice-codec")
    writer.add_name("OmniVoice-Codec-HiggsAudioV2")
    writer.add_uint32("codec.sample_rate", cfg["sample_rate"])
    writer.add_uint32("codec.hop_length", ac["hop_length"])
    writer.add_uint32("codec.n_codebooks", ac["n_codebooks"])
    writer.add_uint32("codec.codebook_size", ac["codebook_size"])
    writer.add_uint32("codec.codebook_dim", ac["codebook_dim"])
    writer.add_uint32("codec.encoder_hidden_size", ac["encoder_hidden_size"])
    writer.add_uint32("codec.decoder_hidden_size", ac["decoder_hidden_size"])
    writer.add_array("codec.upsampling_ratios", list(ac["upsampling_ratios"]))
    writer.add_array("codec.downsampling_ratios", list(ac["downsampling_ratios"]))
    writer.add_uint32("codec.semantic_hidden_size",
                      cfg["semantic_model_config"]["hidden_size"])

    # Decode path only (quantizer + fc2 + acoustic_decoder). The semantic /
    # acoustic_encoder tensors are for voice-clone ENCODE (P5) and some of their
    # names exceed ggml's 63-char limit; include them later behind an encode flag.
    keep_prefixes = ("quantizer.", "fc2.", "acoustic_decoder.")
    n = skipped = 0
    for key, arr in iter_tensors(st):
        if not key.startswith(keep_prefixes):
            skipped += 1
            continue
        # inference decode needs floats only; drop bool/training-only buffers
        if key.endswith(".inited"):
            skipped += 1
            continue
        if arr.dtype == np.bool_ or arr.dtype == np.uint8:
            skipped += 1
            continue
        if len(key) >= 64:
            print(f"  WARN skip long name ({len(key)}): {key}")
            skipped += 1
            continue
        arr = np.ascontiguousarray(arr)
        # CPU ggml_conv_1d/im2col REQUIRES f16 conv kernels; CUDA/Metal accept f32
        # and CUDA's conv_transpose_1d is f32-ONLY. So: f16 conv kernels for the CPU
        # codec (default), f32 conv kernels (--codec-conv-f32) to run the codec on GPU.
        if (not conv_f32) and arr.ndim == 3 and key.endswith(".weight") and arr.dtype == np.float32:
            arr = arr.astype(np.float16)  # conv kernels only; snake .alpha stays f32
        writer.add_tensor(key, arr)
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  wrote {out.name}: {n} tensors ({skipped} non-float skipped)")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=find_default_snapshot(),
                    help="OmniVoice HF snapshot dir (auto-resolves cache)")
    ap.add_argument("--out", type=Path, default=Path("models"),
                    help="output dir for the .gguf files")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="weight dtype for 2D+ matrices (default f32 for parity)")
    ap.add_argument("--part", choices=["all", "generator", "codec"],
                    default="all")
    ap.add_argument("--codec-conv-f32", action="store_true",
                    help="keep codec conv kernels f32 (needed to run the codec on GPU/CUDA)")
    args = ap.parse_args()

    if args.src is None or not Path(args.src).exists():
        print("error: could not resolve --src snapshot dir", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    want_f16 = args.dtype == "f16"
    print(f"src   = {args.src}")
    print(f"out   = {args.out}  dtype={args.dtype}")

    if args.part in ("all", "generator"):
        convert_generator(args.src, args.out / "omnivoice-generator.gguf", want_f16)
    if args.part in ("all", "codec"):
        convert_codec(args.src, args.out / "omnivoice-codec.gguf", want_f16, args.codec_conv_f32)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
