# omnivoice-tts-cpp — a C++/ggml inference engine for OmniVoice

A from-scratch native inference engine for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
(0.6B, 600+ languages, 24 kHz zero-shot TTS), built on [ggml](https://github.com/ggml-org/ggml) —
in the spirit of `vibevoice.cpp` / `parakeet.cpp`. No such engine existed before; this ports the
[omnivoice-rs](https://github.com/FerrisMind/omnivoice-rs) (Candle) reference to ggml.

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

## Architecture (see `MODEL_SPEC.md`, `STAGE0_DESIGN.md`)
- **Backbone**: Qwen3-0.6B, 28 layers, **bidirectional** (non-causal) attention, per-head q/k
  RMSNorm, NEOX RoPE θ=1e6, GQA 16/8, SwiGLU. Reused by both diffusion forwards.
- **Generation**: non-autoregressive masked diffusion — start all-mask, iteratively unmask the
  8×T codebook grid over 32 steps using classifier-free guidance + confidence-ranked scheduling.
- **Codec**: Higgs Audio V2 tokenizer's acoustic path — RVQ dequant → fc2 → DAC decoder
  (conv1d, transposed-conv upsampling ×960, Snake activations) → 24 kHz PCM.

## Build & run
```bash
./scripts/setup.sh                 # fetch ggml, convert weights + tokenizer, build
# end-to-end text -> speech:
./build/tts models/omnivoice-generator.gguf models/omnivoice-codec.gguf \
    models/tokenizer.bin out.wav --text "Hello, this is a test." --lang en
# component tests:
./build/test_tokenizer models/tokenizer.bin
./build/test_qwen3 models/omnivoice-generator.gguf /tmp/h.bin "9707,11,1879,0,151645,198,40,1079"
```

## Key facts (hard-won)
- **Generator must be f32.** With f16, argmax flips cascade over 32 diffusion steps (→ ~7% token
  match). f32 → 100%. Codec conv kernels are f16 (ggml `im2col` requires it); rest f32.
- ggml's default softmax (no mask) IS bidirectional attention — matches OmniVoice's `full_attention`.
- ggml `conv_transpose_1d` has no padding/output_padding; crop `(s+1)/2` from the left to
  reproduce PyTorch `padding=ceil(s/2), output_padding=s%2`.

## Layout
`src/` C++ engine · `tools/` GGUF converter + numpy oracles · `vendor/ggml` (fetched) ·
`reference/omnivoice-rs` (studied, gitignored) · `models/*.gguf` (generated).
```
