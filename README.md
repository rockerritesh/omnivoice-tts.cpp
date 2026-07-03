# omnivoice-tts-cpp — a C++/ggml inference engine for OmniVoice

A from-scratch native inference engine for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
(0.6B, 600+ languages, 24 kHz zero-shot TTS), built on [ggml](https://github.com/ggml-org/ggml) —
in the spirit of `vibevoice.cpp` / `parakeet.cpp`. No such engine existed before; this ports the
[omnivoice-rs](https://github.com/FerrisMind/omnivoice-rs) (Candle) reference to ggml
(mirror fork: [rockerritesh/omnivoice-rs](https://github.com/rockerritesh/omnivoice-rs)).

## Status — COMPLETE: raw text → 24 kHz speech, entirely in C++/ggml

```
text ─▶ [BPE tokenizer + prompt] ─▶ [stage0 diffusion] ─▶ tokens[8,T] ─▶ [DAC codec] ─▶ 24kHz WAV
             ✅ exact ids              ✅ 100% token          ✅ SNR 45–53 dB
```
`./build/tts gen.gguf codec.gguf tokenizer.bin out.wav --text "Hello world." --lang en`
reproduces the reference audio at **cosine 1.00000** for the sample sentence. Runs on **CPU,
NVIDIA CUDA, or Apple Metal** — Tesla T4 CUDA is ~44× faster than CPU on the backbone and still
100% token-exact; Apple M4 Metal is faster than real-time (see Performance).

| Component | C++ file | Test vs reference | Result |
|---|---|---|---|
| Qwen3-0.6B backbone (bidirectional) | `src/test_qwen3.cpp` | vs numpy oracle | **cosine 1.000000** |
| Stage0 masked-diffusion generator | `src/stage0.cpp` | token grid, f32, temps=0 | **100.00% (576/576)** |
| Higgs/DAC acoustic vocoder | `src/test_codec.cpp` | vs reference waveform | **SNR 45.67 dB** |
| Qwen2 byte-level BPE tokenizer | `src/tokenizer.hpp` | ids vs HF (en + Devanagari) | **exact match** |
| **Full text→speech** | `src/tts.cpp` | WAV vs reference | **cosine 1.00000, SNR 53 dB** |

Every stage runs on ggml (CPU) and is numerically validated against omnivoice-rs. The tokenizer
handles Latin, **Devanagari** (नमस्ते → identical ids — relevant for Maithili), digits, and
contractions. Voice cloning (reference-audio encode via HuBERT + Whisper) is the remaining
optional feature; the core zero-shot/voice-design TTS path is complete.

## Multilingual samples (`samples/`)

Generated end-to-end by `./build/tts` (24 kHz mono WAV). OmniVoice supports 600+ languages;
these demo a few, including **Maithili** and **Nepali**:

| Lang | code | file | text |
|---|---|---|---|
| English | `en` | [samples/en.wav](samples/en.wav) | "Hello, this is a multilingual text to speech demo…" |
| Hindi | `hi` | [samples/hi.wav](samples/hi.wav) | "नमस्ते, यह … बहुभाषी वाक् संश्लेषण है।" |
| Maithili | `mai` | [samples/mai.wav](samples/mai.wav) | "प्रणाम, ई एकटा मैथिली वाक् संश्लेषण डेमो अछि…" |
| Nepali | `npi` | [samples/npi.wav](samples/npi.wav) | "नमस्ते, यो … बहुभाषिक वाक् प्रणाली हो।" |
| Chinese | `zh` | [samples/zh.wav](samples/zh.wav) | "你好，这是一个… 多语言语音合成演示。" |
| Spanish | `es` | [samples/es.wav](samples/es.wav) | "Hola, esta es una demostración… de voz multilingüe." |

**Multilingual correctness is verified, not just "it runs":** the C++ output matches the
omnivoice-rs reference bit-exactly (temps=0) for **English (cosine 1.00000)** and
**Maithili (cosine 1.00000, SNR 52 dB)**.

## Performance & backends (CPU / CUDA / Metal)

Two binaries: **`tts`** (plain CPU path, reference-exact) and **`tts_cuda`** (same pipeline via the
ggml-backend API — **CUDA**, **Metal**, or CPU). The Qwen3 diffusion backbone runs on the accelerator;
the cheap codec stays on CPU (`conv_transpose_1d` has no GPU kernel).

Benchmark — "Hello, this is a test…" (2.88 s audio, 32 diffusion steps = 64 backbone forwards):

| Device / backend | dtype | **Stage0 backbone** | Codec | Tokens vs ref | Note |
|---|---|---|---|---|---|
| GCP n1-highmem-4 CPU (4 vCPU) | f32 | ~138 s | 2.0 s | 100% (exact) | same box as the T4 |
| **Tesla T4 · CUDA** | **f32** | **3.15 s** | 2.0 s | **100% (exact)** | **≈44× vs same-box CPU** |
| Tesla T4 · CUDA | f16 | **1.74 s** | 2.0 s | 4.7% † | fastest; tensor cores |
| Apple M4 Pro CPU (12c) | f32 | 19.6 s | 0.39 s | 100% (exact) | consumer CPU |
| **Apple M4 Pro · Metal** | f32 | **2.18 s** | 0.37 s | 83% † | **RTF 0.89 — faster than real-time** |

- **CUDA f32 is reference-exact (100%) *and* ~44× faster** than the same machine's CPU on the backbone.
- **Metal on M4 runs faster than real-time** (RTF 0.89) end-to-end.
- † GPU float reductions aren't bit-identical to CPU, and argmax over near-tied diffusion logits is
  sensitive — so f16 (and, less so, Metal f32) take a *different valid sampling path*, not worse audio.
  The upstream reference likewise treats exact-token parity as diagnostic-only. **Use `tts` (CPU f32)
  when you need bit-exact reproducibility; use `tts_cuda` for speed.**
- The codec (0.37 s on M4) is only slow on the T4 box because that VM's CPU is weak; it's not on the GPU.

### vs official PyTorch (honest)
The official `omnivoice` (PyTorch) engine is **faster, and by a lot on NVIDIA**:

| Device | official torch (fp16) | this engine (f32) | torch advantage |
|---|---|---|---|
| NVIDIA L4 · CUDA | RTF ~0.15 | RTF ~1.85 | **~12×** |
| Apple M4 · MPS/Metal | RTF 0.66 | RTF 0.89 | ~1.35× |

torch's **CUDA** kernels are elite (fused/flash attention, cuDNN, fp16 tensor cores, batched CFG),
so it dominates on NVIDIA; its **MPS** backend is much less tuned, so this engine stays close on Apple.

**Diagnosis (measured):** each backbone forward is **compute-bound, not overhead-bound** — f16 is ~2×
faster than f32 on the T4 (27 vs 49 ms/fwd), which wouldn't happen if launch/alloc overhead dominated.
So the CUDA gap is roughly: **f32-vs-fp16 (~2×) × unbatched-CFG (~2×) × ggml-vs-cuDNN/flash kernels
for this small-seq shape (~2–3×)**.

**Optimizations applied:** the graph is now **built once and reused across all 64 forwards** (was
rebuilt+reallocated each step) — ~10% on GPU, ~15% on CPU, 100% token-parity preserved. **f16 is a
valid fast path** (produces good speech, like torch's fp16; ~2× on CUDA tensor cores). Further levers:
**batched CFG** (~2×, in progress) and **ggml flash-attention**.

**Honest ceiling:** beating PyTorch on a datacenter NVIDIA GPU is *not* a realistic target — cuDNN +
flash-attention are the product of years of kernel engineering. The achievable goal is narrowing the
CUDA gap to ~3×; the engine already **wins on what it's for**: no Python/torch runtime, competitive on
CPU/Apple-Metal, GGUF weights, portable to edge/embedded.

**Positioning:** the goal isn't to beat PyTorch on a datacenter GPU — it's a **dependency-free native
binary** (no Python/torch), competitive on **CPU and Apple Metal**, shipping GGUF weights, runnable
anywhere ggml runs (edge/embedded/Apple/CPU). On NVIDIA, use PyTorch; for a lean portable binary, use this.

## Architecture (see `MODEL_SPEC.md`, `STAGE0_DESIGN.md`)
- **Backbone**: Qwen3-0.6B, 28 layers, **bidirectional** (non-causal) attention, per-head q/k
  RMSNorm, NEOX RoPE θ=1e6, GQA 16/8, SwiGLU. Reused by both diffusion forwards.
- **Generation**: non-autoregressive masked diffusion — start all-mask, iteratively unmask the
  8×T codebook grid over 32 steps using classifier-free guidance + confidence-ranked scheduling.
- **Codec**: Higgs Audio V2 tokenizer's acoustic path — RVQ dequant → fc2 → DAC decoder
  (conv1d, transposed-conv upsampling ×960, Snake activations) → 24 kHz PCM.

## Prerequisites
- A C++17 compiler + **CMake ≥ 3.16** (macOS: Xcode Command Line Tools; Linux: gcc/clang)
- **git**, and [**uv**](https://github.com/astral-sh/uv) (used to run the Python tooling with
  pinned deps — no venv setup needed)
- **~24 GB RAM** for CPU f32 inference; **~7 GB disk** (3.5 GB model download + 2.7 GB GGUFs)
- Internet access for the first run (fetches ggml + the OmniVoice weights from Hugging Face)

## Build & run
```bash
git clone https://github.com/rockerritesh/omnivoice-tts.cpp
cd omnivoice-tts.cpp
./scripts/setup.sh          # downloads model (~3.5GB), fetches ggml, converts weights, builds (CPU)
```
`setup.sh` builds the **CPU** binary. For GPU, reconfigure with one flag and build `tts_cuda`:
```bash
# NVIDIA CUDA  (set the arch: T4=75, A100=80, L4/Ada=89, H100=90)
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DOMNI_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target tts_cuda -j

# Apple Metal (MPS) — macOS / Apple Silicon
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DOMNI_METAL=ON
cmake --build build --target tts_cuda -j

# CPU-only backend build of the same binary (no flags)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --target tts_cuda -j
```
Run — `tts` (CPU, reference-exact) and `tts_cuda` (CUDA/Metal/CPU) take identical args:
```bash
# reference-exact CPU:
./build/tts       models/omnivoice-generator.gguf models/omnivoice-codec.gguf \
    models/tokenizer.bin out.wav --text "Hello, this is a test." --lang en
# GPU-accelerated (auto-uses CUDA or Metal if built with it, else CPU):
./build/tts_cuda  models/omnivoice-generator.gguf models/omnivoice-codec.gguf \
    models/tokenizer.bin out.wav --text "Hello, this is a test." --lang en

# other languages (code from k2-fsa/OmniVoice, e.g. hi, mai, npi, zh, es):
./build/tts models/omnivoice-generator.gguf models/omnivoice-codec.gguf \
    models/tokenizer.bin mai.wav --text "प्रणाम, ई मैथिली टेस्ट अछि।" --lang mai

# optional flags: --duration SEC  --num-step 32  --guidance 2.0  --instruct "..."
# component tests:
./build/test_tokenizer models/tokenizer.bin
```
If you already have the `k2-fsa/OmniVoice` snapshot in `~/.cache/huggingface`, the download in
step 2 is a no-op. Weights are pulled automatically; nothing model-related is committed here.

## Key facts (hard-won)
- **Generator must be f32.** With f16, argmax flips cascade over 32 diffusion steps (→ ~7% token
  match). f32 → 100%. Codec conv kernels are f16 (ggml `im2col` requires it); rest f32.
- ggml's default softmax (no mask) IS bidirectional attention — matches OmniVoice's `full_attention`.
- ggml `conv_transpose_1d` has no padding/output_padding; crop `(s+1)/2` from the left to
  reproduce PyTorch `padding=ceil(s/2), output_padding=s%2`.

For max GPU throughput, convert an f16 generator (`python tools/convert_omnivoice_to_gguf.py
--part generator --dtype f16`) — ~1.6× faster on the T4 via tensor cores, but it takes a different
sampling path (see the † note). Keep the f32 generator for reference-exact output.

## Layout
`src/` C++ engine (`tts.cpp` CPU-exact · `tts_cuda.cpp` CUDA/Metal/CPU via ggml-backend) ·
`tools/` GGUF converter + numpy oracles · `vendor/ggml` (fetched) · `models/*.gguf` (generated) ·
`samples/` demos.

## License
Original code here: **PolyForm Noncommercial License 1.0.0** — free for research, education, and
other **noncommercial** use; **not for commercial use** (see [LICENSE](LICENSE)). Third-party
dependencies and the OmniVoice model weights carry their own licenses — see [NOTICE](NOTICE).
Model weights are not included; download them from [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).
