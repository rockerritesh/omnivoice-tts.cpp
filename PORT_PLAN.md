# OmniVoice C++ Inference Engine — Port Plan

Goal: a native C++/ggml inference engine for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
(0.6B, 600+ languages, zero-shot voice cloning, 24 kHz), in the spirit of
`vibevoice.cpp` / `parakeet.cpp`.

## Strategy: port the Rust (Candle) engine, not the Python

No C++/ggml engine exists yet (checked the official community-projects list). The
[omnivoice-rs](https://github.com/FerrisMind/omnivoice-rs) Candle port is our **blueprint** —
it has already reverse-engineered every hard part and ships phase-by-phase behavior contracts
(`docs/contracts/`) and a golden-tensor harness (`tools/python_reference/`). Candle ops map
almost 1:1 onto ggml, so we port Rust → C++ rather than re-deriving from PyTorch.

Cloned for study at `reference/omnivoice-rs/` (gitignored, not vendored).

## Architecture (confirmed from the Rust source)

Two-stage, **non-autoregressive diffusion LM** over discrete audio tokens:

| Component | What it is | Rust file | C++ target |
|---|---|---|---|
| Frontend | text norm, language id, duration, voice-design prompt | `frontend/*.rs` | plain C++ |
| Reference ASR | Whisper (turbo) transcribes the clone reference | `asr.rs` | **whisper.cpp** |
| Audio tokenizer (encode) | HuBERT (semantic) + **DAC** (acoustic, Conv1d+Snake1d) | `audio_tokenizer_{hubert,dac}.rs` | ggml |
| Stage0 | Qwen3-0.6B diffusion/unmask loop → coarse tokens, w/ CFG | `stage0_qwen3.rs`, `stage0_model.rs`, `stage0_loop.rs` | **llama.cpp** Qwen3 graph + custom loop |
| Stage1 | Qwen3 refinement → all 8 codebooks | `stage1_model.rs` | llama.cpp graph |
| Stage1 decoder | DAC decoder: 8 codebooks (vocab 1025) → 24 kHz PCM | `stage1_decoder.rs` | ggml (Conv1d + Snake) |
| Postprocess | trim/join chunks → WAV | `postprocess.rs` | plain C++ |

Key numeric details already extracted:
- **8 audio codebooks**, vocab 1025 each; embeddings summed via `codebook_layer_offsets`.
- Stage0 loop = flow-matching timesteps `build_timesteps(t_start,t_end,num_step,t_shift)` +
  confidence-based `build_unmask_schedules` + classifier-free guidance (packed `2*batch`).
- DAC = `AcousticEncoder`/decoder of Conv1d blocks with `acoustic_downsampling_ratios` and
  `Snake1d` activation (`x + sin²(αx)/α`) — a trivial ggml custom op.

## Reuse from the ggml ecosystem
- **llama.cpp** already implements the Qwen3 transformer → reuse for Stage0/Stage1 graphs.
- **whisper.cpp** → reference-audio ASR path, drop-in.
- **DAC in ggml** already exists in TTS.cpp / Parler-TTS.cpp ports → crib the codec decoder.
- ggml provides `conv_1d`; Snake is a custom unary op.

## Phases (each validated against golden tensors before moving on)

- **P0 — Reference & golden tensors.** ✅ DONE (oracle). Built omnivoice-cli (Metal, 1m49s), ran
  `infer` CPU/f32 seed 1234 → `golden/baseline_en_cpu.wav` (mono 24 kHz, 73,920 samples = 3.08 s).
  HF snapshot `999c3324…`. Intermediate-tensor dumps (Stage0 `(C,T)` grid etc.) deferred to P2/P3,
  where layer-by-layer validation actually needs them — premature before any C++ exists.
- **P1 — Weight conversion.** ✅ DONE. `tools/convert_omnivoice_to_gguf.py` emits
  `omnivoice-generator.gguf` (Qwen3→llama.cpp names + audio tensors + contracts KV, 1.1 GB f16)
  and `omnivoice-codec.gguf` (DAC decoder + RVQ + configs, 386 MB f16). Verified via GGUFReader.
  Full tensor map: [MODEL_SPEC.md](MODEL_SPEC.md).
- **P2 — Qwen3 backbone in ggml.** ✅ DONE & TESTED. `src/test_qwen3.cpp` on vendored ggml;
  bidirectional (no-mask) + causal paths both match the numpy oracle `tools/qwen3_ref.py`
  (row cosine 1.000000, mean abs diff 0.0033, f16 precision). Retires the biggest arch risk.
- **P3 — Stage0 diffusion loop** (timesteps, unmask schedule, CFG) → token grid parity.
  🔬 FULLY DESIGNED — algorithm reverse-engineered & documented in [STAGE0_DESIGN.md](STAGE0_DESIGN.md).
  Key finding: **deterministic at `class_temperature=0, position_temperature=0`** → bit-exact
  testable. Reuses the P2 backbone graph. C++ implementation (`src/stage0.cpp`) is the remaining
  build; until the frontend (P5) exists it consumes a dumped prepared-batch so the diffusion math
  is validated in isolation against a temps=0 golden grid.
- **P4 — Stage1 + DAC decoder.** ✅ DONE & TESTED. `src/test_codec.cpp`: RVQ dequant + fc2 +
  acoustic DAC decoder (conv1d, conv_transpose_1d with crop math, dilated residual units, Snake).
  Decode uses ONLY the acoustic path (no semantic). Golden `(tokens[8,72]→waveform[69120])`
  dumped from an instrumented omnivoice-rs; C++ matches at **cosine 0.999986, SNR 45.67 dB**.
  Conv kernels f16 (ggml im2col requirement); everything else f32.
- **P5 — Voice cloning** (HuBERT+DAC encode + whisper.cpp ASR) and long-form chunking.
- **P6 — CLI** (`omnivoice-cpp infer --text --language --output`) + optional server.

## Confirmed checkpoint layout & contracts (from `omnivoice.artifacts.json`)

HF repo `k2-fsa/OmniVoice` auto-downloads to the HF cache. Two safetensors files:

**`model.safetensors`** (the generator / Qwen3 LLM), tensor prefixes:
- `llm.*` — Qwen3-0.6B backbone
- `audio_embeddings.*` — the 8 codebook embedding tables (vocab 1025 each)
- `audio_heads.*` — the 8 output heads predicting codebook logits
- `codebook_layer_offsets` — a constant buffer (not a weight; the C++ code must reproduce it)

**`audio_tokenizer/model.safetensors`** (the codec), tensor prefixes:
- `semantic_model.*` — HuBERT (semantic tokens for clone reference)
- `acoustic_encoder.*` — DAC encoder (Conv1d + Snake1d)
- `acoustic_decoder.*` — **DAC decoder → 24 kHz PCM** (the vocoder)
- `quantizer.*` — RVQ codebooks

Text tokenizer: standard HF `tokenizer.json` + `chat_template.jinja` (Qwen3).

**Runtime contracts:** 8 codebooks · vocab 1025 · mask_id 1024 · valid token ids 0–1023 ·
24 kHz · hop_length 960 · **frame_rate 25 Hz** (so 1 s audio = 25 frames × 8 codebooks = 200 tokens).

## Open questions to resolve in P0
- Exact HuBERT variant and how semantic+acoustic tokens combine in the prompt.
- Whether Stage1 is a full second Qwen3 pass or a lighter head.
- DAC config (downsampling ratios, hidden sizes) — read from checkpoint config.
