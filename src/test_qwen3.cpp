// P2 test: OmniVoice Qwen3 backbone forward pass in ggml (CPU, bidirectional).
// Loads omnivoice-generator.gguf, runs a forward on a fixed token sequence,
// and writes the final hidden states as raw f32 for parity checking against
// tools/qwen3_ref.py (the numpy oracle).
//
// Usage: test_qwen3 <generator.gguf> <out.bin> <tok0,tok1,...> [causal]
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_tensor * must(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    return t;
}

static uint32_t kv_u32(gguf_context * g, const char * key, uint32_t dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_u32(g, id);
}
static float kv_f32(gguf_context * g, const char * key, float dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_f32(g, id);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <generator.gguf> <out.bin> <t0,t1,...> [causal]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    bool causal = (argc > 4 && std::string(argv[4]) == "causal");

    std::vector<int32_t> ids;
    { std::string s = argv[3], cur;
      for (char c : s) { if (c == ',') { ids.push_back(atoi(cur.c_str())); cur.clear(); } else cur += c; }
      if (!cur.empty()) ids.push_back(atoi(cur.c_str())); }
    const int T = (int) ids.size();

    // --- load weights ---
    ggml_context * wctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &wctx };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }

    const int   n_layer = (int) kv_u32(g, "qwen3.block_count", 28);
    const int   n_embd  = (int) kv_u32(g, "qwen3.embedding_length", 1024);
    const int   n_head  = (int) kv_u32(g, "qwen3.attention.head_count", 16);
    const int   n_kv    = (int) kv_u32(g, "qwen3.attention.head_count_kv", 8);
    const int   hd      = (int) kv_u32(g, "qwen3.attention.key_length", 128);
    const float eps     = kv_f32(g, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    const float theta   = kv_f32(g, "qwen3.rope.freq_base", 1000000.0f);
    printf("model: L=%d n_embd=%d n_head=%d n_kv=%d hd=%d eps=%g theta=%g | T=%d mask=%s\n",
           n_layer, n_embd, n_head, n_kv, hd, eps, theta, T, causal ? "causal" : "full");

    // --- compute context ---
    ggml_init_params cp = { (size_t) 2 * 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(cp);

    ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    for (int i = 0; i < T; i++) {
        ((int32_t *) tok->data)[i] = ids[i];
        ((int32_t *) pos->data)[i] = i;
    }

    // optional causal mask [T,T] (f32): 0 on/below diag, -inf above
    ggml_tensor * mask = nullptr;
    if (causal) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
        float * m = (float *) mask->data;
        for (int q = 0; q < T; q++)
            for (int k = 0; k < T; k++)
                m[q * T + k] = (k <= q) ? 0.0f : -INFINITY;
    }

    ggml_tensor * cur = ggml_get_rows(ctx, must(wctx, "token_embd.weight"), tok); // [n_embd,T]
    if (cur->type != GGML_TYPE_F32) cur = ggml_cast(ctx, cur, GGML_TYPE_F32);

    const float kq_scale = 1.0f / sqrtf((float) hd);

    for (int il = 0; il < n_layer; il++) {
        std::string p = "blk." + std::to_string(il) + ".";
        ggml_tensor * inp = cur;

        // --- attention ---
        ggml_tensor * x = ggml_rms_norm(ctx, cur, eps);
        x = ggml_mul(ctx, x, must(wctx, (p + "attn_norm.weight").c_str()));

        ggml_tensor * q = ggml_mul_mat(ctx, must(wctx, (p + "attn_q.weight").c_str()), x);
        ggml_tensor * k = ggml_mul_mat(ctx, must(wctx, (p + "attn_k.weight").c_str()), x);
        ggml_tensor * v = ggml_mul_mat(ctx, must(wctx, (p + "attn_v.weight").c_str()), x);

        q = ggml_reshape_3d(ctx, q, hd, n_head, T);
        k = ggml_reshape_3d(ctx, k, hd, n_kv,   T);
        v = ggml_reshape_3d(ctx, v, hd, n_kv,   T);

        // per-head q/k RMSNorm (over head_dim), pre-RoPE
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), must(wctx, (p + "attn_q_norm.weight").c_str()));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), must(wctx, (p + "attn_k_norm.weight").c_str()));

        q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd,T,n_head]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd,T,n_kv]
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                        // [T,T,n_head]
        kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, 0.0f);               // mask=NULL -> bidirectional

        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [T,hd,n_kv]
        ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);                       // [hd,T,n_head]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));            // [hd,n_head,T]
        ggml_tensor * o = ggml_cont_2d(ctx, kqv, hd * n_head, T);            // [n_embd_q,T]
        o = ggml_mul_mat(ctx, must(wctx, (p + "attn_output.weight").c_str()), o); // [n_embd,T]

        cur = ggml_add(ctx, inp, o);
        ggml_tensor * inp2 = cur;

        // --- MLP (SwiGLU) ---
        ggml_tensor * y = ggml_rms_norm(ctx, cur, eps);
        y = ggml_mul(ctx, y, must(wctx, (p + "ffn_norm.weight").c_str()));
        ggml_tensor * gate = ggml_mul_mat(ctx, must(wctx, (p + "ffn_gate.weight").c_str()), y);
        ggml_tensor * up   = ggml_mul_mat(ctx, must(wctx, (p + "ffn_up.weight").c_str()), y);
        gate = ggml_silu(ctx, gate);
        y = ggml_mul(ctx, gate, up);
        y = ggml_mul_mat(ctx, must(wctx, (p + "ffn_down.weight").c_str()), y);
        cur = ggml_add(ctx, inp2, y);
    }

    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_mul(ctx, cur, must(wctx, "output_norm.weight")); // [n_embd,T]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    // cur is [n_embd, T] (ne0=n_embd). Print stats + write row-major [T,n_embd].
    const float * h = (const float *) cur->data;
    const int ne0 = (int) cur->ne[0]; // n_embd
    printf("hidden ne=[%lld,%lld]\n", (long long) cur->ne[0], (long long) cur->ne[1]);
    printf("hidden[row0,:6]  =");
    for (int j = 0; j < 6; j++) printf(" %.6f", h[0 * ne0 + j]); printf("\n");
    printf("hidden[row%d,:6] =", T - 1);
    for (int j = 0; j < 6; j++) printf(" %.6f", h[(T - 1) * ne0 + j]); printf("\n");
    double s = 0, s2 = 0, amax = 0; long n = (long) T * ne0;
    for (long i = 0; i < n; i++) { double x = h[i]; s += x; s2 += x * x; if (fabs(x) > amax) amax = fabs(x); }
    printf("mean=%.6f std=%.6f absmax=%.6f\n", s / n, sqrt(s2 / n - (s / n) * (s / n)), amax);

    FILE * f = fopen(out_path, "wb");
    fwrite(h, sizeof(float), n, f);
    fclose(f);
    printf("wrote %s (%ld floats, layout [T=%d, n_embd=%d])\n", out_path, n, T, ne0);

    ggml_free(ctx);
    gguf_free(g);
    ggml_free(wctx);
    return 0;
}
