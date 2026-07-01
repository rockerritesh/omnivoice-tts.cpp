#!/usr/bin/env bash
# Fetch external deps and convert weights. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. ggml (core tensor lib + gguf loader) — pinned clone under vendor/
if [ ! -d vendor/ggml ]; then
  git clone --depth 1 https://github.com/ggml-org/ggml vendor/ggml
fi

# 2. Download the OmniVoice checkpoint (~3.5 GB) into the HF cache. Skipped if
#    already present. Set HF_TOKEN if the repo ever requires authentication.
uv run --with huggingface_hub python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('k2-fsa/OmniVoice'))"

# 3. Convert the OmniVoice checkpoint to GGUF (auto-resolves the HF cache).
#    --dtype f32: the generator MUST be f32 — f16 argmax flips cascade over the
#    32 diffusion steps and wreck the token grid. Codec conv kernels are forced
#    to f16 internally (ggml im2col requirement); its other weights stay f32.
uv run --with gguf --with safetensors --with numpy \
  python tools/convert_omnivoice_to_gguf.py --dtype f32 --out models
uv run --with tokenizers python tools/export_tokenizer.py --out models/tokenizer.bin

# 4. Build.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
echo
echo "setup done. binaries in build/. Try:"
echo "  ./build/tts models/omnivoice-generator.gguf models/omnivoice-codec.gguf \\"
echo "      models/tokenizer.bin out.wav --text \"Hello, this is a test.\" --lang en"
