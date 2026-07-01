# OmniVoice — Complete Model Spec (for the C++/ggml port)

Derived directly from the downloaded `k2-fsa/OmniVoice` checkpoint
(snapshot `999c3324…`). Two safetensors files, all F32.

## 1. Generator — `model.safetensors` (313 tensors, ~612M params)

The diffusion LM. Qwen3-0.6B backbone + audio embedding/head tables.

### 1a. Qwen3 backbone (`llm.*`, ~596M)
Standard Qwen3, prefix `llm.` (note: `llm.` not `llm.model.`), **tied embeddings, no lm_head**.

| tensor | shape | notes |
|---|---|---|
| `llm.embed_tokens.weight` | [151676, 1024] | text vocab; tied |
| `llm.layers.{0..27}.input_layernorm.weight` | [1024] | RMSNorm |
| `llm.layers.{i}.self_attn.q_proj.weight` | [2048, 1024] | 16 heads × 128 |
| `llm.layers.{i}.self_attn.k_proj.weight` | [1024, 1024] | 8 KV heads × 128 |
| `llm.layers.{i}.self_attn.v_proj.weight` | [1024, 1024] | |
| `llm.layers.{i}.self_attn.o_proj.weight` | [1024, 2048] | |
| `llm.layers.{i}.self_attn.q_norm.weight` | [128] | Qwen3 per-head RMSNorm |
| `llm.layers.{i}.self_attn.k_norm.weight` | [128] | Qwen3 per-head RMSNorm |
| `llm.layers.{i}.post_attention_layernorm.weight` | [1024] | |
| `llm.layers.{i}.mlp.gate_proj.weight` | [3072, 1024] | SwiGLU |
| `llm.layers.{i}.mlp.up_proj.weight` | [3072, 1024] | |
| `llm.layers.{i}.mlp.down_proj.weight` | [1024, 3072] | |
| `llm.norm.weight` | [1024] | final RMSNorm |

Config: hidden 1024, 28 layers, 16 heads / 8 KV, head_dim 128, ffn 3072, silu,
rms_eps 1e-6, rope_theta 1e6, vocab 151676. **All layers `full_attention` → NON-CAUSAL
(bidirectional) attention. No KV cache reuse across diffusion steps in the usual AR sense.**

### 1b. Audio I/O (OmniVoice-specific)
| tensor | shape | notes |
|---|---|---|
| `audio_embeddings.weight` | [8200, 1024] | 8 codebooks × 1025 vocab, stacked |
| `audio_heads.weight` | [8200, 1024] | output logits, sliced 8 × 1025 |
| `codebook_layer_offsets` | I64 [8] | row offset per codebook (0,1025,2050,…) |

- **Embed:** for codebook c, token id t → row `offset[c]+t` of `audio_embeddings`; sum the
  8 codebook embeddings at each audio position, add to the LLM input stream.
- **Head:** hidden → `audio_heads` [8200] → split into 8 chunks of 1025 → per-codebook logits.
- `audio_codebook_weights` = [8,8,6,6,4,4,2,2] (per-codebook weighting; used by
  `layer_penalty_factor` in the sampler).
- Contracts: 8 codebooks · vocab 1025 · mask_id 1024 · valid ids 0–1023 · 24 kHz · hop 960 · 25 Hz.

## 2. Audio tokenizer / codec — `audio_tokenizer/model.safetensors` (527 tensors)
`HiggsAudioV2TokenizerModel` = HuBERT semantic + DAC acoustic + RVQ. Decode path is the vocoder.

| group | #T | params | role |
|---|---|---|---|
| `semantic_model.*` | 210 | 94.4M | HuBERT (768d, 12 layers) — encode ref audio (clone) |
| `acoustic_encoder.*` | 110 | 51.3M | DAC encoder — encode ref audio (clone) |
| `acoustic_decoder.*` | 110 | 20.2M | **DAC decoder → 24 kHz PCM (the vocoder)** |
| `quantizer.*` | 64 | 2.1M | RVQ: 8× `codebook.embed`[1024,64] + project_in/out |
| `encoder_semantic.*` / `decoder_semantic.*` | 13/14 | ~31M | semantic conv path |
| `fc`,`fc1`,`fc2` | 6 | 2.1M | proj 1024→1024 / 1024→768 / 1024→256 |

DAC decoder unit (repeats per upsampling stage, ratios [8,5,4,2,3] → 960× up):
- `acoustic_decoder.block.{i}.conv_t1.{weight,bias}` — transposed conv (upsample)
- `...res_unitN.conv1.weight` [C,C,7], `conv2.weight` [C,C,1] — dilated residual
- `...res_unitN.snake{1,2}.alpha` [1,C,1] — **Snake activation** `x + (1/α)·sin²(αx)`
ggml needs: `conv_1d`, `conv_transpose_1d`, and a custom Snake unary op.

## 3. GGUF conversion plan (`convert_omnivoice_to_gguf.py`)
Emit **two** GGUF files (loaded by separate ggml contexts in the C++ engine):
1. `omnivoice-generator.gguf` — Qwen3 tensors renamed to llama.cpp convention
   (`token_embd.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_k_norm.weight`, …) so
   llama.cpp's loader/graph can be reused; **plus** `audio_embeddings`, `audio_heads`,
   `codebook_layer_offsets` as extra tensors; contracts baked into KV metadata.
2. `omnivoice-codec.gguf` — acoustic_decoder + quantizer (+ encoder/semantic for cloning),
   kept in native names (custom loader, not llama.cpp).

For the text-only TTS MVP (P2–P4) only the generator + codec **decoder** + quantizer are
needed; HuBERT/acoustic_encoder/Whisper are P5 (voice cloning).
