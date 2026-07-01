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
reproduces the reference audio at **cosine 1.00000** for the sample sentence.

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

## Performance

Apple M4 Pro, CPU, f32, "Hello, this is a test…" (2.88 s of audio, 32 diffusion steps):

| Engine | wall time | RTF | runtime deps |
|---|---|---|---|
| **this (C++/ggml)** | **22.8 s** | ~7.9× | ggml only (single native binary) |
| omnivoice-rs (Rust/Candle) | 22.2 s | ~7.7× | Candle |
| official (PyTorch) | — | — | torch + torchaudio |

Honest take: on CPU it's **on par** with the Candle reference (not faster yet) — the win is a
lean, dependency-free native binary with GGUF weights, not raw speed. Headroom for large gains
is untapped: Metal backend (currently CPU-only), graph/KV reuse instead of 64 full recomputes,
and quantization. The upstream model reports RTF 0.025 (40× real-time) on GPU.

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
./scripts/setup.sh          # downloads model (~3.5GB), fetches ggml, converts weights, builds

# end-to-end text -> speech (writes out.wav, 24 kHz mono):
./build/tts models/omnivoice-generator.gguf models/omnivoice-codec.gguf \
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

## Layout
`src/` C++ engine · `tools/` GGUF converter + numpy oracles · `vendor/ggml` (fetched) ·
`reference/omnivoice-rs` (studied, gitignored) · `models/*.gguf` (generated) · `samples/` demos.

## License
Original code here: **PolyForm Noncommercial License 1.0.0** — free for research, education, and
other **noncommercial** use; **not for commercial use** (see [LICENSE](LICENSE)). Third-party
dependencies and the OmniVoice model weights carry their own licenses — see [NOTICE](NOTICE).
Model weights are not included; download them from [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).
