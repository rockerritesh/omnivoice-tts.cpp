# Stage0 Diffusion Loop — Design (P3)

Reverse-engineered from `stage0_model.rs` / `stage0_loop.rs`. This is the text→token-grid
generator: a **non-autoregressive masked diffusion** over the 8-codebook audio grid, driven by
the Qwen3 backbone (already ported & tested in P2).

## Testability (important)
The sampler is **fully deterministic when `class_temperature=0` AND `position_temperature=0`**:
- `class_temperature=0` → token prediction is plain `argmax` (no gumbel/RNG).
- `position_temperature=0` → unmask-position selection is plain top-k by score (no gumbel/RNG).

So P3 is bit-exact testable: run reference `infer --class-temperature 0 --position-temperature 0`
(+ `OMNI_DUMP`) to dump a golden token grid, and compare the C++ grid element-wise. (The default
run uses 0.0/5.0 — stochastic positions — which is why the *default* path is only perceptual.)

## Inputs (the "prepared batch", from the frontend + `pack_cfg_batch`)
CFG doubles the batch: rows `[0..B)` = conditional, `[B..2B)` = unconditional.
- `input_ids`      : i64 `[2B, C=8, L]`  — text token ids on text positions, audio-codebook ids
  (init = mask_id 1024) on the `target_len` audio positions.
- `audio_mask`     : bool/int `[2B, ?, L]` — marks which positions are audio (get summed codebook
  embeds) vs text (get text embed from `llm.embed_tokens`).
- `attention_mask` : `[2B, L]` — 1 for real tokens, 0 for padding → additive bias.
- `tokens_init`    : i64 `[B, C, target_len]` — all mask_id.
- `cond_lens[b]`, `target_lens[b]` — audio region is `input_ids[b, :, cond_len-target_len : cond_len]`.

## Embedding construction (`model.forward`, to confirm when coding)
`inputs_embeds[pos] = text_embed(input_ids[pos])` for text positions; for audio positions,
`sum_c audio_embeddings[ offset[c] + input_ids[b,c,pos] ]` where `offset=[0,1025,…]`. The audio_mask
selects which rule applies. Backbone = Qwen3 (bidirectional, P2). Output head = `audio_heads`
`[8200,1024]` → per position, logits `[8, 1025]` (split by codebook).

## The loop (`generate_deterministic`)
```
timesteps = build_timesteps(0, 1, num_step+1, t_shift)          # t_shift=0.1 default
schedules[b][step] = build_unmask_schedules(target_lens, C=8, timesteps, num_step)
layer_penalty[c]   = c * layer_penalty_factor                    # factor=5.0 default; shape [1,8,1]
tokens      = tokens_init                                        # [B,8,target_len], all mask
batch_ids   = input_ids                                         # [2B,8,L]
for step in 0..num_step:                                         # num_step=32 default
    logits = backbone.forward(batch_ids, audio_mask, attn_mask) # [2B,8,L,1025]
    for b in 0..B:
        c_logits = logits[b,      :, cond-target : cond, :]      # conditional, audio region
        u_logits = logits[B+b,    :, 0 : target,        :]      # unconditional
        # classifier-free guidance in log-space:
        lc = log_softmax(c_logits); lu = log_softmax(u_logits)
        guided = log_softmax( lc + g*(lc - lu) )                # g=guidance_scale=2.0 default
        guided[..., mask_id] = -inf                             # never predict the mask token
        pred[b,c,t]  = argmax_v guided[b,c,t,v]                 # class_temperature=0
        conf[b,c,t]  = max_v  guided[b,c,t,v]
        # unmask schedule[b][step] positions with highest (conf - layer_penalty[c]):
        score = conf - layer_penalty                            # [8, target]
        score[ tokens[b] != mask_id ] = -inf                    # only fill still-masked slots
        idx  = argsort(flatten(score), desc)[: schedule[b][step]]
        tokens[b].flat[idx] = pred[b].flat[idx]                 # scatter predictions
    # write updated tokens back into BOTH cond and uncond copies of batch_ids audio region
final grid = tokens                                             # [B,8,target_len]
```
`build_timesteps(a,b,n,shift)`: `t_i = a + (b-a)*i/n`, then `shift*t/(1+(shift-1)*t)`.
`build_unmask_schedules`: per step, unmask `ceil(total_mask * (t[step+1]-t[step]))` (last step: all
remaining), where `total_mask = target_len * 8`.

## C++ port plan (`src/stage0.cpp`, reuses P2 backbone graph)
1. Dump the prepared batch + a temps=0 golden token grid from an instrumented omnivoice-rs.
2. Backbone forward extended: build `inputs_embeds` (text vs summed-audio via audio_mask),
   run the P2 Qwen3 graph over the full `[2B,L]` batch, apply `audio_heads` → `[2B,8,L,1025]`.
3. CFG + argmax + confidence (plain C++ over the audio region logits).
4. Unmask top-k by (conf − layer_penalty) among masked slots; scatter.
5. Loop num_step; emit `[8, target_len]` → feed `test_codec` → WAV.
6. Test: element-wise vs golden grid (temps=0) → then WAV vs oracle.

Frontend (tokenizer.json BPE + chat template + duration→target_len) is P5; until then the C++
loop consumes the dumped prepared batch so the diffusion math can be validated in isolation.
```
