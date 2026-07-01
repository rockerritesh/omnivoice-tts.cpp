// P4 test: OmniVoice acoustic codec (Higgs/DAC) decoder in ggml (CPU).
// tokens [C=8, T] -> RVQ dequant -> fc2 -> acoustic_decoder -> 24kHz PCM.
// Validates against golden/codec/raw_audio_0.bin dumped from omnivoice-rs.
//
// Usage: test_codec <codec.gguf> <tokens.bin> <out.bin>
//   tokens.bin layout: [i64 C][i64 T][C*T i64 ids]  (row-major, C-major)
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

static ggml_tensor * must(ggml_context * c, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(c, n.c_str());
    if (!t) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
    return t;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <codec.gguf> <tokens.bin> <out.bin>\n", argv[0]); return 1; }

    // --- read tokens ---
    FILE * tf = fopen(argv[2], "rb");
    if (!tf) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    int64_t C = 0, T = 0;
    if (fread(&C, 8, 1, tf) != 1 || fread(&T, 8, 1, tf) != 1) { return 1; }
    std::vector<int64_t> toks((size_t) C * T);
    if (fread(toks.data(), 8, toks.size(), tf) != toks.size()) { return 1; }
    fclose(tf);
    printf("tokens: C=%lld T=%lld\n", (long long) C, (long long) T);

    // --- load codec weights ---
    ggml_context * wctx = nullptr;
    gguf_init_params gp = { false, &wctx };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }

    const int strides[5] = { 8, 5, 4, 2, 3 };

    ggml_init_params cp = { (size_t) 6 * 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * eps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    ((float *) eps->data)[0] = 1e-9f;

    // helpers -----------------------------------------------------------------
    auto conv1d = [&](ggml_tensor * x, const std::string & name, int pad, int dil) {
        ggml_tensor * w = must(wctx, name + ".weight");
        ggml_tensor * y = ggml_conv_1d(ctx, w, x, 1, pad, dil);      // [OL,OC,N]
        ggml_tensor * b = ggml_get_tensor(wctx, (name + ".bias").c_str());
        if (b) y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
        return y;
    };
    auto snake = [&](ggml_tensor * x, const std::string & name) {   // x + sin(a*x)^2/(a+eps)
        ggml_tensor * a = must(wctx, name + ".alpha");              // [1,C,1]
        ggml_tensor * s2 = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, a)));
        ggml_tensor * den = ggml_add(ctx, a, eps);
        return ggml_add(ctx, x, ggml_div(ctx, s2, den));
    };
    auto convt = [&](ggml_tensor * x, const std::string & name, int stride) {
        ggml_tensor * w = must(wctx, name + ".weight");            // [2s, C/2, C]
        int64_t Lin = x->ne[0];
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1); // [Lin*s+s, OC,1,1]
        int64_t oc = y->ne[1];
        int64_t keep = Lin * stride;
        int64_t crop = (stride + 1) / 2;                          // PyTorch padding=ceil(s/2)
        ggml_tensor * v = ggml_view_2d(ctx, y, keep, oc, y->nb[1], crop * y->nb[0]);
        v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), keep, oc, 1);
        ggml_tensor * b = must(wctx, name + ".bias");
        return ggml_add(ctx, v, ggml_reshape_3d(ctx, b, 1, oc, 1));
    };
    auto res_unit = [&](ggml_tensor * x, const std::string & name, int dil) {
        ggml_tensor * y = snake(x, name + ".snake1");
        y = conv1d(y, name + ".conv1", 3 * dil, dil);             // k7, pad=3*dil
        y = snake(y, name + ".snake2");
        y = conv1d(y, name + ".conv2", 0, 1);                     // k1
        return ggml_add(ctx, y, x);
    };

    // --- RVQ dequant: sum_c project_out_c(embed_c[ids_c]) ---
    ggml_tensor * quant = nullptr;
    for (int c = 0; c < (int) C; c++) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        for (int t = 0; t < (int) T; t++) ((int32_t *) ids->data)[t] = (int32_t) toks[(size_t) c * T + t];
        std::string q = "quantizer.quantizers." + std::to_string(c) + ".";
        ggml_tensor * emb  = ggml_get_rows(ctx, must(wctx, q + "codebook.embed"), ids);   // [64,T]
        ggml_tensor * proj = ggml_mul_mat(ctx, must(wctx, q + "project_out.weight"), emb); // [1024,T]
        proj = ggml_add(ctx, proj, ggml_reshape_2d(ctx, must(wctx, q + "project_out.bias"), proj->ne[0], 1));
        quant = quant ? ggml_add(ctx, quant, proj) : proj;
    }

    // --- fc2 (1024->256), to length-major [T,256,1] ---
    ggml_tensor * h = ggml_mul_mat(ctx, must(wctx, "fc2.weight"), quant);   // [256,T]
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, must(wctx, "fc2.bias"), h->ne[0], 1));
    h = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, h)), T, h->ne[0], 1); // [T,256,1]

    // --- acoustic decoder ---
    h = conv1d(h, "acoustic_decoder.conv1", 3, 1);                          // [T,1024,1]
    for (int i = 0; i < 5; i++) {
        std::string p = "acoustic_decoder.block." + std::to_string(i);
        h = snake(h, p + ".snake1");
        h = convt(h, p + ".conv_t1", strides[i]);
        h = res_unit(h, p + ".res_unit1", 1);
        h = res_unit(h, p + ".res_unit2", 3);
        h = res_unit(h, p + ".res_unit3", 9);
    }
    h = snake(h, "acoustic_decoder.snake1");
    h = conv1d(h, "acoustic_decoder.conv2", 3, 1);                          // [L,1,1]

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, h);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    const int64_t L = h->ne[0];
    const float * out = (const float *) h->data;
    printf("output waveform: L=%lld (%.3f s, %lld/frame)\n", (long long) L, L / 24000.0, (long long) (L / T));
    double s = 0, mn = 1e9, mx = -1e9;
    for (int64_t i = 0; i < L; i++) { s += out[i] * out[i]; if (out[i] < mn) mn = out[i]; if (out[i] > mx) mx = out[i]; }
    printf("rms=%.5f min=%.5f max=%.5f\n", sqrt(s / L), mn, mx);

    FILE * of = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), L, of);
    fclose(of);
    printf("wrote %s (%lld floats)\n", argv[3], (long long) L);

    ggml_free(ctx); gguf_free(g); ggml_free(wctx);
    return 0;
}
