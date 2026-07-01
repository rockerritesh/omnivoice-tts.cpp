#!/usr/bin/env bash
# Fetch external deps and convert weights. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. ggml (core tensor lib + gguf loader) — pinned clone under vendor/
if [ ! -d vendor/ggml ]; then
  git clone --depth 1 https://github.com/ggml-org/ggml vendor/ggml
fi

# 2. Convert the OmniVoice checkpoint to GGUF (auto-resolves the HF cache).
#    Requires the k2-fsa/OmniVoice snapshot in ~/.cache/huggingface.
uv run --with gguf --with safetensors --with numpy \
  python tools/convert_omnivoice_to_gguf.py --dtype f16 --out models

# 3. Build.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
echo "setup done. binaries in build/"
