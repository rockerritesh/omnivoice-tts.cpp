// P3: OmniVoice Stage0 masked-diffusion generator in ggml (CPU).
// Consumes a prepared batch dumped from omnivoice-rs (OMNI_DUMP_BATCH) and runs
// the deterministic (temps=0) diffusion loop -> [8, target_len] token grid.
// See STAGE0_DESIGN.md. Reuses the P2 Qwen3 backbone (bidirectional).
//
// Usage: stage0 <generator.gguf> <batch_dir> <out_tokens.bin>
//   batch_dir holds input_ids.bin, audio_mask.bin, meta.txt (from OMNI_DUMP_BATCH)
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

struct Arr { std::vector<int64_t> dims; std::vector<float> data; };

static Arr load_arr(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    int64_t nd; if (fread(&nd, 8, 1, f) != 1) exit(1);
    Arr a; a.dims.resize(nd);
    if ((int64_t) fread(a.dims.data(), 8, nd, f) != nd) exit(1);
    int64_t n = 1; for (auto d : a.dims) n *= d;
    a.data.resize(n);
    if ((int64_t) fread(a.data.data(), 4, n, f) != n) exit(1);
    fclose(f);
    return a;
}

static ggml_tensor * must(ggml_context * c, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(c, n.c_str());
    if (!t) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
    return t;
}
static uint32_t kv_u32(gguf_context * g, const char * k, uint32_t d) {
    int64_t id = gguf_find_key(g, k); return id < 0 ? d : gguf_get_val_u32(g, id);
}
static float kv_f32(gguf_context * g, const char * k, float d) {
    int64_t id = gguf_find_key(g, k); return id < 0 ? d : gguf_get_val_f32(g, id);
}

struct Hparams { int n_layer, n_embd, n_head, n_kv, hd; float eps, theta; };

// Run the Qwen3 backbone (bidirectional) on a prebuilt inputs_embeds [n_embd, S],
// then audio_heads -> returns logits laid out [1025, 8, S] as a flat vector.
static std::vector<float> forward_audio_logits(
        ggml_context * wctx, const Hparams & hp, int n_cb, int vocab,
        const std::vector<int32_t> & text_ids,      // [S]
        const std::vector<int32_t> & aud_ids,        // [8*S], codebook-major, already shifted
        const std::vector<float> & amask,            // [S] 0/1
        int S) {
    ggml_init_params cp = { (size_t) 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(cp);

    // --- inputs_embeds = where(amask, sum_c audio_emb[aud], text_emb[text]) ---
    ggml_tensor * tids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    memcpy(tids->data, text_ids.data(), S * sizeof(int32_t));
    ggml_tensor * temb = ggml_get_rows(ctx, must(wctx, "token_embd.weight"), tids); // [H,S]
    if (temb->type != GGML_TYPE_F32) temb = ggml_cast(ctx, temb, GGML_TYPE_F32);

    ggml_tensor * aemb_w = must(wctx, "audio_embeddings.weight"); // [H,8200]
    ggml_tensor * aud_sum = nullptr;
    for (int c = 0; c < n_cb; c++) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        int32_t * p = (int32_t *) ids->data;
        for (int t = 0; t < S; t++) p[t] = aud_ids[(size_t) c * S + t];
        ggml_tensor * e = ggml_get_rows(ctx, aemb_w, ids);       // [H,S]
        if (e->type != GGML_TYPE_F32) e = ggml_cast(ctx, e, GGML_TYPE_F32);
        aud_sum = aud_sum ? ggml_add(ctx, aud_sum, e) : e;
    }
    // mask row [1,S] broadcast over H
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S);
    ggml_tensor * mm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S);
    for (int t = 0; t < S; t++) { ((float*)m->data)[t] = amask[t]; ((float*)mm->data)[t] = 1.0f - amask[t]; }
    ggml_tensor * cur = ggml_add(ctx, ggml_mul(ctx, aud_sum, m), ggml_mul(ctx, temb, mm)); // [H,S]

    // --- Qwen3 backbone (same as P2, bidirectional / no mask) ---
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    for (int t = 0; t < S; t++) ((int32_t*)pos->data)[t] = t;
    const int hd = hp.hd, nh = hp.n_head, nkv = hp.n_kv;
    const float scale = 1.0f / sqrtf((float) hd);
    for (int il = 0; il < hp.n_layer; il++) {
        std::string p = "blk." + std::to_string(il) + ".";
        ggml_tensor * inp = cur;
        ggml_tensor * x = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), must(wctx, p+"attn_norm.weight"));
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, must(wctx, p+"attn_q.weight"), x), hd, nh, S);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, must(wctx, p+"attn_k.weight"), x), hd, nkv, S);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, must(wctx, p+"attn_v.weight"), x), hd, nkv, S);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.eps), must(wctx, p+"attn_q_norm.weight"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.eps), must(wctx, p+"attn_k_norm.weight"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1,0,1,0,0);
        k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1,0,1,0,0);
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0,2,1,3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0,2,1,3));
        ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kp, qp), nullptr, scale, 0.0f);
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1,2,0,3));
        ggml_tensor * kqv = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vp, kq), 0,2,1,3));
        ggml_tensor * o = ggml_mul_mat(ctx, must(wctx, p+"attn_output.weight"), ggml_cont_2d(ctx, kqv, hd*nh, S));
        cur = ggml_add(ctx, inp, o);
        ggml_tensor * inp2 = cur;
        ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), must(wctx, p+"ffn_norm.weight"));
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, must(wctx, p+"ffn_gate.weight"), y));
        y = ggml_mul(ctx, gate, ggml_mul_mat(ctx, must(wctx, p+"ffn_up.weight"), y));
        cur = ggml_add(ctx, inp2, ggml_mul_mat(ctx, must(wctx, p+"ffn_down.weight"), y));
    }
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), must(wctx, "output_norm.weight")); // [H,S]
    // audio heads: [H,8200] x [H,S] -> [8200,S]
    ggml_tensor * logits = ggml_mul_mat(ctx, must(wctx, "audio_heads.weight"), cur);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    std::vector<float> out((size_t) vocab * n_cb * S);
    memcpy(out.data(), logits->data, out.size() * sizeof(float)); // [1025,8,S] layout
    ggml_free(ctx);
    return out;
}

static std::vector<float> build_timesteps(float a, float b, int n, float shift) {
    std::vector<float> ts(n + 1);
    for (int i = 0; i <= n; i++) { float t = a + (b - a) * i / n; ts[i] = (shift * t) / (1 + (shift - 1) * t); }
    return ts;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <generator.gguf> <batch_dir> <out.bin>\n", argv[0]); return 1; }
    std::string dir = argv[2];

    // --- load prepared batch ---
    Arr input_ids = load_arr(dir + "/input_ids.bin");   // [2, 8, L]
    Arr audio_mask = load_arr(dir + "/audio_mask.bin");  // [2, L] (or [2,1,L])
    int B2 = input_ids.dims[0], C = input_ids.dims[1], L = input_ids.dims[2];
    int Lm = audio_mask.dims.back();
    // meta
    int num_step = 32, cond_len = 0, target_len = 0; float guidance = 2.0f, t_shift = 0.1f, lpf = 5.0f;
    { FILE * f = fopen((dir + "/meta.txt").c_str(), "r"); char line[256];
      while (f && fgets(line, sizeof line, f)) {
        int v; float fv;
        if (sscanf(line, "cond_lens=[%d", &v) == 1) cond_len = v;
        else if (sscanf(line, "target_lens=[%d", &v) == 1) target_len = v;
        else if (sscanf(line, "num_step=%d", &v) == 1) num_step = v;
        else if (sscanf(line, "guidance_scale=%f", &fv) == 1) guidance = fv;
        else if (sscanf(line, "t_shift=%f", &fv) == 1) t_shift = fv;
        else if (sscanf(line, "layer_penalty_factor=%f", &fv) == 1) lpf = fv;
      } if (f) fclose(f); }
    printf("batch: B2=%d C=%d L=%d | cond_len=%d target_len=%d num_step=%d g=%.2f t_shift=%.2f lpf=%.1f\n",
           B2, C, L, cond_len, target_len, num_step, guidance, t_shift, lpf);

    // --- load weights + hparams ---
    ggml_context * wctx = nullptr; gguf_init_params gp = { false, &wctx };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "load fail\n"); return 1; }
    Hparams hp {
        (int) kv_u32(g, "qwen3.block_count", 28), (int) kv_u32(g, "qwen3.embedding_length", 1024),
        (int) kv_u32(g, "qwen3.attention.head_count", 16), (int) kv_u32(g, "qwen3.attention.head_count_kv", 8),
        (int) kv_u32(g, "qwen3.attention.key_length", 128),
        kv_f32(g, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f), kv_f32(g, "qwen3.rope.freq_base", 1e6f),
    };
    const int vocab = (int) kv_u32(g, "omnivoice.audio_vocab_size", 1025);
    const int mask_id = (int) kv_u32(g, "omnivoice.audio_mask_id", 1024);
    std::vector<int> offset(C); for (int c = 0; c < C; c++) offset[c] = c * vocab;

    auto idx3 = [&](const Arr & a, int b, int c, int t) { return a.data[((size_t) b * a.dims[1] + c) * a.dims[2] + t]; };

    // working input_ids (mutable), and tokens grid
    std::vector<int32_t> ids0(C * L), ids1(C * L);
    for (int c = 0; c < C; c++) for (int t = 0; t < L; t++) {
        ids0[c * L + t] = (int32_t) llround(idx3(input_ids, 0, c, t));
        ids1[c * L + t] = (int32_t) llround(idx3(input_ids, 1, c, t));
    }
    std::vector<float> am0(cond_len), am1(target_len);
    for (int t = 0; t < cond_len; t++) am0[t] = audio_mask.data[0 * Lm + t];
    for (int t = 0; t < target_len; t++) am1[t] = audio_mask.data[1 * Lm + t];

    std::vector<int32_t> tokens(C * target_len, mask_id);

    // timesteps + unmask schedule
    std::vector<float> ts = build_timesteps(0, 1, num_step + 1, t_shift);
    // build_unmask_schedules expects timesteps length num_step+2; ref passes num_step+1 -> pads?
    // Reproduce: schedule[step] = ceil(total*(ts[step+1]-ts[step])), last = remaining.
    int total = target_len * C, remaining = total;
    std::vector<int> sched(num_step);
    for (int s = 0; s < num_step; s++) {
        int amt = (s == num_step - 1) ? remaining
                  : (int) std::ceil(total * (ts[s + 1] - ts[s]));
        amt = std::min(amt, remaining); sched[s] = amt; remaining -= amt;
    }

    auto text_ids_of = [&](std::vector<int32_t> & ids, int S) {
        std::vector<int32_t> t(S); for (int i = 0; i < S; i++) t[i] = ids[0 * L + i]; return t;
    };
    auto aud_ids_of = [&](std::vector<int32_t> & ids, std::vector<float> & am, int S) {
        std::vector<int32_t> a((size_t) C * S);
        for (int c = 0; c < C; c++) for (int t = 0; t < S; t++) {
            int32_t raw = (am[t] > 0.5f) ? ids[c * L + t] : 0;
            a[c * S + t] = raw + offset[c];
        }
        return a;
    };

    for (int step = 0; step < num_step; step++) {
        if (sched[step] == 0) continue;
        // cond forward (length cond_len), uncond forward (length target_len)
        auto cl = forward_audio_logits(wctx, hp, C, vocab, text_ids_of(ids0, cond_len), aud_ids_of(ids0, am0, cond_len), am0, cond_len);
        auto ul = forward_audio_logits(wctx, hp, C, vocab, text_ids_of(ids1, target_len), aud_ids_of(ids1, am1, target_len), am1, target_len);
        if (step == 0 && getenv("STAGE0_DUMP_LOGITS")) {
            // dump cond logits [1025,8,cond_len] (layout data[(pos*C+c)*vocab+v])
            FILE * f = fopen((dir + "/cpp_step0_cond.bin").c_str(), "wb");
            fwrite(cl.data(), 4, cl.size(), f); fclose(f);
            f = fopen((dir + "/cpp_step0_uncond.bin").c_str(), "wb");
            fwrite(ul.data(), 4, ul.size(), f); fclose(f);
        }
        // logits layout [v, c, t]: index (t*C + c)*vocab + v
        int aud_start = cond_len - target_len;
        std::vector<int32_t> pred((size_t) C * target_len);
        std::vector<float> conf((size_t) C * target_len);
        for (int c = 0; c < C; c++) for (int t = 0; t < target_len; t++) {
            const float * cv = &cl[((size_t)(aud_start + t) * C + c) * vocab];
            const float * uv = &ul[((size_t) t * C + c) * vocab];
            // log_softmax of each
            auto lse = [&](const float * z){ float mx=-1e30f; for(int v=0;v<vocab;v++) mx=std::max(mx,z[v]); double s=0; for(int v=0;v<vocab;v++) s+=exp(z[v]-mx); return mx+(float)log(s); };
            float lc_z = lse(cv), lu_z = lse(uv);
            // guided_v = (1+g)*lc - g*lu ; then log_softmax(guided); mask id -> -inf; argmax/max
            static thread_local std::vector<float> guided; guided.resize(vocab);
            float gmx = -1e30f;
            for (int v = 0; v < vocab; v++) {
                float lc = cv[v] - lc_z, lu = uv[v] - lu_z;
                guided[v] = (1.0f + guidance) * lc - guidance * lu;
                gmx = std::max(gmx, guided[v]);
            }
            double gs = 0; for (int v = 0; v < vocab; v++) gs += exp(guided[v] - gmx);
            float gz = gmx + (float) log(gs);
            int best = 0; float bestval = -1e30f;
            for (int v = 0; v < vocab; v++) {
                float ls = guided[v] - gz;
                if (v == mask_id) ls = -INFINITY;
                if (ls > bestval) { bestval = ls; best = v; }
            }
            pred[c * target_len + t] = best; conf[c * target_len + t] = bestval;
        }
        // unmask top sched[step] by (conf - c*lpf) among masked slots
        std::vector<int> cand;
        for (int c = 0; c < C; c++) for (int t = 0; t < target_len; t++)
            if (tokens[c * target_len + t] == mask_id) cand.push_back(c * target_len + t);
        auto score = [&](int idx){ int c = idx / target_len; return conf[idx] - c * lpf; };
        std::sort(cand.begin(), cand.end(), [&](int a, int b){ return score(a) > score(b); });
        int k = std::min((int) sched[step], (int) cand.size());
        for (int i = 0; i < k; i++) {
            int idx = cand[i]; int c = idx / target_len, t = idx % target_len;
            tokens[idx] = pred[idx];
            ids0[c * L + (aud_start + t)] = pred[idx];   // update cond audio region
            ids1[c * L + t] = pred[idx];                 // update uncond region
        }
        int filled = 0; for (auto v : tokens) if (v != mask_id) filled++;
        printf("step %2d: unmask %4d  (%d/%d filled)\n", step, k, filled, total);
    }

    // write tokens grid [C, target_len] as i64
    FILE * of = fopen(argv[3], "wb");
    int64_t cc = C, tt = target_len; fwrite(&cc, 8, 1, of); fwrite(&tt, 8, 1, of);
    for (int i = 0; i < C * target_len; i++) { int64_t v = tokens[i]; fwrite(&v, 8, 1, of); }
    fclose(of);
    printf("wrote %s [%d,%d]\n", argv[3], C, target_len);
    gguf_free(g); ggml_free(wctx);
    return 0;
}
