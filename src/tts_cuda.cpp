// GPU-capable OmniVoice TTS: same pipeline as tts.cpp but the Qwen3 diffusion
// backbone runs through the ggml-backend API (CUDA if built with GGML_USE_CUDA,
// else CPU). The cheap codec stays on the plain CPU compute path (its
// conv_transpose_1d has no CUDA kernel). Numerically identical to tts.cpp on the
// CPU backend; use it to benchmark GPU vs CPU.
//
// Usage: same as tts.cpp:
//   tts_cuda <gen.gguf> <codec.gguf> <tok.bin> <out.wav> --text "..." [--lang en] ...
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#ifdef OMNI_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef OMNI_USE_METAL
#include "ggml-metal.h"
#endif
#include "tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

static ggml_tensor * must(ggml_context * c, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(c, n.c_str());
    if (!t) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
    return t;
}
static uint32_t kv_u32(gguf_context * g, const char * k, uint32_t d) { int64_t i = gguf_find_key(g, k); return i < 0 ? d : gguf_get_val_u32(g, i); }
static float   kv_f32(gguf_context * g, const char * k, float d)    { int64_t i = gguf_find_key(g, k); return i < 0 ? d : gguf_get_val_f32(g, i); }

struct Hparams { int n_layer, n_embd, n_head, n_kv, hd; float eps, theta; };

static ggml_backend_t make_backend() {
#ifdef OMNI_USE_CUDA
    {
        ggml_backend_t cb = ggml_backend_cuda_init(0);
        if (cb) { fprintf(stderr, "[backend] CUDA\n"); return cb; }
        fprintf(stderr, "[backend] CUDA init failed -> CPU\n");
    }
#endif
#ifdef OMNI_USE_METAL
    {
        ggml_backend_t mb = ggml_backend_metal_init();
        if (mb) { fprintf(stderr, "[backend] Metal\n"); return mb; }
        fprintf(stderr, "[backend] Metal init failed -> CPU\n");
    }
#endif
    ggml_backend_t b = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(b, 8);
    fprintf(stderr, "[backend] CPU\n");
    return b;
}

// Load all GGUF tensors onto `backend`, copying data from the file.
static ggml_context * load_weights(const char * path, ggml_backend_t backend,
                                   ggml_backend_buffer_t * buf_out, gguf_context ** gguf_out) {
    ggml_context * mctx = nullptr;
    gguf_init_params p{ /*no_alloc*/ true, &mctx };
    gguf_context * g = gguf_init_from_file(path, p);
    if (!g) { fprintf(stderr, "gguf load fail: %s\n", path); exit(1); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(mctx, backend);
    FILE * f = fopen(path, "rb");
    size_t data_off = gguf_get_data_offset(g);
    std::vector<char> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        ggml_tensor * t = ggml_get_tensor(mctx, gguf_get_tensor_name(g, i));
        size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        fseek(f, (long)(data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "read fail %s\n", t->name); exit(1); }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
    }
    fclose(f);
    *buf_out = buf; *gguf_out = g;
    return mctx;
}

// Persistent backbone graph, built ONCE per sequence length and reused across all
// diffusion steps. This is the key optimization: the old code rebuilt+reallocated
// the 28-layer graph on every one of the 64 forwards, which dominated GPU time.
struct Fwd {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * tids = nullptr, * mA = nullptr, * mT = nullptr, * pos = nullptr, * logits = nullptr;
    std::vector<ggml_tensor*> aids;
    int S = 0, n_cb = 0, vocab = 0;
};

static Fwd * fwd_build(ggml_backend_t backend, ggml_context * wctx, const Hparams & hp,
                       int n_cb, int vocab, const std::vector<float> & amask, int S) {
    Fwd * f = new Fwd(); f->S = S; f->n_cb = n_cb; f->vocab = vocab; f->aids.resize(n_cb);
    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    f->ctx = ggml_init({ meta, nullptr, /*no_alloc*/ true });
    ggml_context * ctx = f->ctx;

    f->tids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S); ggml_set_input(f->tids);
    for (int c = 0; c < n_cb; c++) { f->aids[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S); ggml_set_input(f->aids[c]); }
    f->mA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S); ggml_set_input(f->mA);
    f->mT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S); ggml_set_input(f->mT);
    f->pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);   ggml_set_input(f->pos);

    ggml_tensor * temb = ggml_get_rows(ctx, must(wctx, "token_embd.weight"), f->tids);
    if (temb->type != GGML_TYPE_F32) temb = ggml_cast(ctx, temb, GGML_TYPE_F32);
    ggml_tensor * aemb_w = must(wctx, "audio_embeddings.weight");
    ggml_tensor * aud_sum = nullptr;
    for (int c = 0; c < n_cb; c++) {
        ggml_tensor * e = ggml_get_rows(ctx, aemb_w, f->aids[c]);
        if (e->type != GGML_TYPE_F32) e = ggml_cast(ctx, e, GGML_TYPE_F32);
        aud_sum = aud_sum ? ggml_add(ctx, aud_sum, e) : e;
    }
    ggml_tensor * cur = ggml_add(ctx, ggml_mul(ctx, aud_sum, f->mA), ggml_mul(ctx, temb, f->mT));

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
        q = ggml_rope_ext(ctx, q, f->pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1,0,1,0,0);
        k = ggml_rope_ext(ctx, k, f->pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1,0,1,0,0);
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
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), must(wctx, "output_norm.weight"));
    f->logits = ggml_mul_mat(ctx, must(wctx, "audio_heads.weight"), cur);
    ggml_set_output(f->logits);

    f->gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(f->gf, f->logits);
    f->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(f->galloc, f->gf);

    // mask + positions are constant across diffusion steps -> set once.
    std::vector<float> ma(S), mt(S); for (int t=0;t<S;t++){ ma[t]=amask[t]; mt[t]=1.0f-amask[t]; }
    ggml_backend_tensor_set(f->mA, ma.data(), 0, (size_t)S*sizeof(float));
    ggml_backend_tensor_set(f->mT, mt.data(), 0, (size_t)S*sizeof(float));
    std::vector<int32_t> pv(S); for (int t=0;t<S;t++) pv[t]=t;
    ggml_backend_tensor_set(f->pos, pv.data(), 0, (size_t)S*sizeof(int32_t));
    return f;
}

// Update token inputs and recompute the persistent graph -> logits [1025,8,S] into `out`.
static void fwd_run(ggml_backend_t backend, Fwd * f, const std::vector<int32_t> & text_ids,
                    const std::vector<int32_t> & aud_ids, std::vector<float> & out) {
    ggml_backend_tensor_set(f->tids, text_ids.data(), 0, (size_t)f->S*sizeof(int32_t));
    for (int c = 0; c < f->n_cb; c++)
        ggml_backend_tensor_set(f->aids[c], aud_ids.data() + (size_t)c*f->S, 0, (size_t)f->S*sizeof(int32_t));
    ggml_backend_graph_compute(backend, f->gf);
    out.resize((size_t)f->vocab*f->n_cb*f->S);
    ggml_backend_tensor_get(f->logits, out.data(), 0, out.size()*sizeof(float));
}

// ---- codec decode on CPU (unchanged from tts.cpp) ----
// Codec decode on CPU. ggml's CUDA conv_transpose_1d kernel is unoptimized and ~40x
// SLOWER than CPU for this upsampling (measured 14s on T4 vs 0.4s on a normal CPU),
// so the codec always runs on CPU regardless of the backbone backend.
static std::vector<float> decode_codec(ggml_context * wctx, const std::vector<int64_t> & toks, int C, int T) {
    const int strides[5] = { 8, 5, 4, 2, 3 };
    ggml_init_params cp = { (size_t) 6ULL * 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * eps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1); ((float*)eps->data)[0] = 1e-9f;

    auto conv1d = [&](ggml_tensor * x, const std::string & n, int pad, int dil) {
        ggml_tensor * y = ggml_conv_1d(ctx, must(wctx, n+".weight"), x, 1, pad, dil);
        ggml_tensor * b = ggml_get_tensor(wctx, (n+".bias").c_str());
        if (b) y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
        return y; };
    auto snake = [&](ggml_tensor * x, const std::string & n) {
        ggml_tensor * a = must(wctx, n+".alpha");
        ggml_tensor * s2 = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, a)));
        return ggml_add(ctx, x, ggml_div(ctx, s2, ggml_add(ctx, a, eps))); };
    auto convt = [&](ggml_tensor * x, const std::string & n, int s) {
        int64_t Lin = x->ne[0];
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, must(wctx, n+".weight"), x, s, 0, 1);
        int64_t oc = y->ne[1], keep = Lin * s, crop = (s+1)/2;
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, y, keep, oc, y->nb[1], crop*y->nb[0])), keep, oc, 1);
        return ggml_add(ctx, v, ggml_reshape_3d(ctx, must(wctx, n+".bias"), 1, oc, 1)); };
    auto res_unit = [&](ggml_tensor * x, const std::string & n, int dil) {
        ggml_tensor * y = snake(x, n+".snake1"); y = conv1d(y, n+".conv1", 3*dil, dil);
        y = snake(y, n+".snake2"); y = conv1d(y, n+".conv2", 0, 1); return ggml_add(ctx, y, x); };
    ggml_tensor * quant = nullptr;
    for (int c = 0; c < C; c++) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        for (int t = 0; t < T; t++) ((int32_t*)ids->data)[t] = (int32_t) toks[(size_t)c*T+t];
        std::string q = "quantizer.quantizers." + std::to_string(c) + ".";
        ggml_tensor * proj = ggml_mul_mat(ctx, must(wctx, q+"project_out.weight"), ggml_get_rows(ctx, must(wctx, q+"codebook.embed"), ids));
        proj = ggml_add(ctx, proj, ggml_reshape_2d(ctx, must(wctx, q+"project_out.bias"), proj->ne[0], 1));
        quant = quant ? ggml_add(ctx, quant, proj) : proj;
    }
    ggml_tensor * h = ggml_mul_mat(ctx, must(wctx, "fc2.weight"), quant);
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, must(wctx, "fc2.bias"), h->ne[0], 1));
    h = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, h)), T, h->ne[0], 1);
    h = conv1d(h, "acoustic_decoder.conv1", 3, 1);
    for (int i = 0; i < 5; i++) {
        std::string p = "acoustic_decoder.block." + std::to_string(i);
        h = snake(h, p+".snake1"); h = convt(h, p+".conv_t1", strides[i]);
        h = res_unit(h, p+".res_unit1", 1); h = res_unit(h, p+".res_unit2", 3); h = res_unit(h, p+".res_unit3", 9);
    }
    h = snake(h, "acoustic_decoder.snake1");
    h = conv1d(h, "acoustic_decoder.conv2", 3, 1);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, h);
    ggml_graph_compute_with_ctx(ctx, gf, 8);
    std::vector<float> out(h->ne[0]);
    memcpy(out.data(), h->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// duration estimator (ported char-weights)
static float char_weight(int cp) {
    if ((cp>='A'&&cp<='Z')||(cp>='a'&&cp<='z')) return 1.0f;
    if (cp==' ') return 0.2f; if (cp>='0'&&cp<='9') return 3.5f; if (cp<0x80) return 0.5f;
    if ((cp>=0x0900&&cp<=0x0903)||(cp>=0x093A&&cp<=0x094F)||(cp>=0x0951&&cp<=0x0957)||(cp>=0x0962&&cp<=0x0963)) return 0.0f;
    if (cp>=0x0900&&cp<=0x0DFF) return 1.8f; if (cp>=0x0E00&&cp<=0x0EFF) return 1.5f;
    if (cp>=0x4E00&&cp<=0x9FFF) return 3.0f; if (cp>=0x0400&&cp<=0x04FF) return 1.0f; if (cp>=0x0600&&cp<=0x06FF) return 1.5f;
    return 1.0f;
}
static std::vector<int> utf8_cps(const std::string & t) {
    std::vector<int> v; size_t i=0; while(i<t.size()){ unsigned char c=t[i]; int cp,len;
        if(c<0x80){cp=c;len=1;}else if((c>>5)==0x6){cp=c&0x1F;len=2;}else if((c>>4)==0xE){cp=c&0x0F;len=3;}else{cp=c&0x07;len=4;}
        for(int k=1;k<len&&i+k<t.size();k++)cp=(cp<<6)|(t[i+k]&0x3F); v.push_back(cp); i+=len; } return v;
}
static int estimate_target_len(const std::string & text) {
    float rw=0; for(int cp:utf8_cps("Nice to meet you.")) rw+=char_weight(cp);
    float tw=0; for(int cp:utf8_cps(text)) tw+=char_weight(cp);
    float est = tw/(rw/25.0f); if(est<50.0f) est=50.0f*powf(est/50.0f,1.0f/3.0f);
    return std::max(1,(int)est);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <gen.gguf> <codec.gguf> <tok.bin> <out.wav> --text \"...\" [--lang en] [--duration S] [--num-step N] [--guidance G] [--t-shift T] [--layer-penalty L] [--dump-tokens f]\n", argv[0]); return 1; }
    std::string gen=argv[1], codec=argv[2], tokpath=argv[3], out=argv[4];
    std::string text, lang="en", instruct="None", dump_tokens; float duration=-1, guidance=2.0f, t_shift=0.1f, lpf=5.0f; int num_step=32;
    for (int i=5;i<argc;i++){ std::string a=argv[i]; auto nx=[&]{ return (i+1<argc)?argv[++i]:""; };
        if(a=="--text")text=nx(); else if(a=="--lang")lang=nx(); else if(a=="--instruct")instruct=nx();
        else if(a=="--duration")duration=atof(nx()); else if(a=="--num-step")num_step=atoi(nx());
        else if(a=="--guidance")guidance=atof(nx()); else if(a=="--t-shift")t_shift=atof(nx());
        else if(a=="--layer-penalty")lpf=atof(nx()); else if(a=="--dump-tokens")dump_tokens=nx(); }
    if (text.empty()) { fprintf(stderr, "--text required\n"); return 1; }

    BpeTokenizer tk; if (!tk.load(tokpath)) { fprintf(stderr,"tok load fail\n"); return 1; }
    std::vector<int> prompt; auto add=[&](const std::string&sp){ prompt.push_back(tk.token_to_id(sp)); };
    add("<|lang_start|>"); for(int id:tk.encode(lang)) prompt.push_back(id); add("<|lang_end|>");
    add("<|instruct_start|>"); for(int id:tk.encode(instruct)) prompt.push_back(id); add("<|instruct_end|>");
    add("<|text_start|>"); for(int id:tk.encode(text)) prompt.push_back(id); add("<|text_end|>");
    int text_len=(int)prompt.size();
    int target_len=(duration>0)?std::max(1,(int)lroundf(duration*25.0f)):estimate_target_len(text);
    int cond_len=text_len+target_len;
    printf("text_tokens=%d target_len=%d (%.2fs) cond_len=%d num_step=%d\n", text_len,target_len,target_len/25.0,cond_len,num_step);

    ggml_backend_t backend = make_backend();
    ggml_backend_buffer_t gbuf; gguf_context * gg;
    ggml_context * gwctx = load_weights(gen.c_str(), backend, &gbuf, &gg);
    Hparams hp{ (int)kv_u32(gg,"qwen3.block_count",28),(int)kv_u32(gg,"qwen3.embedding_length",1024),
        (int)kv_u32(gg,"qwen3.attention.head_count",16),(int)kv_u32(gg,"qwen3.attention.head_count_kv",8),
        (int)kv_u32(gg,"qwen3.attention.key_length",128),kv_f32(gg,"qwen3.attention.layer_norm_rms_epsilon",1e-6f),kv_f32(gg,"qwen3.rope.freq_base",1e6f) };
    const int C=(int)kv_u32(gg,"omnivoice.num_audio_codebooks",8), vocab=(int)kv_u32(gg,"omnivoice.audio_vocab_size",1025), mask_id=(int)kv_u32(gg,"omnivoice.audio_mask_id",1024);
    std::vector<int> offset(C); for(int c=0;c<C;c++) offset[c]=c*vocab;

    int L=cond_len;
    std::vector<int32_t> ids0((size_t)C*L,0), ids1((size_t)C*target_len,mask_id);
    for(int t=0;t<text_len;t++) ids0[t]=prompt[t];
    for(int c=0;c<C;c++)for(int t=text_len;t<L;t++) ids0[c*L+t]=mask_id;
    std::vector<float> am0(L,0.0f), am1(target_len,1.0f); for(int t=text_len;t<L;t++) am0[t]=1.0f;
    std::vector<float> ts(num_step+2); for(int i=0;i<=num_step+1;i++){ float t=(float)i/(num_step+1); ts[i]=(t_shift*t)/(1+(t_shift-1)*t); }
    int total=target_len*C, remaining=total; std::vector<int> sched(num_step);
    for(int s=0;s<num_step;s++){ int amt=(s==num_step-1)?remaining:(int)std::ceil(total*(ts[s+1]-ts[s])); amt=std::min(amt,remaining); sched[s]=amt; remaining-=amt; }
    std::vector<int32_t> tokens((size_t)C*target_len,mask_id);
    auto text_ids_of=[&](std::vector<int32_t>&ids,int Sq){ std::vector<int32_t> t(Sq); for(int i=0;i<Sq;i++) t[i]=ids[i]; return t; };
    auto aud_ids_of=[&](std::vector<int32_t>&ids,std::vector<float>&am,int Sq,int rowlen){ std::vector<int32_t> a((size_t)C*Sq);
        for(int c=0;c<C;c++)for(int t=0;t<Sq;t++){ int32_t raw=(am[t]>0.5f)?ids[c*rowlen+t]:0; a[c*Sq+t]=raw+offset[c]; } return a; };

    // Build the cond + uncond backbone graphs ONCE (reused across all steps).
    Fwd * fc = fwd_build(backend, gwctx, hp, C, vocab, am0, cond_len);
    Fwd * fu = fwd_build(backend, gwctx, hp, C, vocab, am1, target_len);
    std::vector<float> cl, ul;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int step=0;step<num_step;step++){
        if(sched[step]==0) continue;
        fwd_run(backend, fc, text_ids_of(ids0,cond_len), aud_ids_of(ids0,am0,cond_len,L), cl);
        fwd_run(backend, fu, text_ids_of(ids1,target_len), aud_ids_of(ids1,am1,target_len,target_len), ul);
        int aud_start=cond_len-target_len;
        std::vector<int32_t> pred((size_t)C*target_len); std::vector<float> conf((size_t)C*target_len);
        for(int c=0;c<C;c++)for(int t=0;t<target_len;t++){
            const float* cv=&cl[((size_t)(aud_start+t)*C+c)*vocab]; const float* uv=&ul[((size_t)t*C+c)*vocab];
            auto lse=[&](const float*z){ float mx=-1e30f; for(int v=0;v<vocab;v++) mx=std::max(mx,z[v]); double s=0; for(int v=0;v<vocab;v++) s+=exp(z[v]-mx); return mx+(float)log(s); };
            float lcz=lse(cv),luz=lse(uv); std::vector<float> guided(vocab); float gmx=-1e30f;
            for(int v=0;v<vocab;v++){ float lc=cv[v]-lcz,lu=uv[v]-luz; guided[v]=(1.0f+guidance)*lc-guidance*lu; gmx=std::max(gmx,guided[v]); }
            double gs=0; for(int v=0;v<vocab;v++) gs+=exp(guided[v]-gmx); float gz=gmx+(float)log(gs);
            int best=0; float bv=-1e30f; for(int v=0;v<vocab;v++){ float ls=guided[v]-gz; if(v==mask_id) ls=-INFINITY; if(ls>bv){bv=ls;best=v;} }
            pred[c*target_len+t]=best; conf[c*target_len+t]=bv;
        }
        std::vector<int> cand; for(int c=0;c<C;c++)for(int t=0;t<target_len;t++) if(tokens[c*target_len+t]==mask_id) cand.push_back(c*target_len+t);
        auto score=[&](int idx){ return conf[idx]-(idx/target_len)*lpf; };
        std::sort(cand.begin(),cand.end(),[&](int a,int b){return score(a)>score(b);});
        int k=std::min((int)sched[step],(int)cand.size());
        for(int i=0;i<k;i++){ int idx=cand[i],c=idx/target_len,t=idx%target_len; tokens[idx]=pred[idx]; ids0[c*L+(aud_start+t)]=pred[idx]; ids1[c*target_len+t]=pred[idx]; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double stage0_s = std::chrono::duration<double>(t1-t0).count();
    printf("stage0: %d tokens in %.2fs (%.1f ms/forward)\n", C*target_len, stage0_s, stage0_s*1000/(num_step*2));

    if (!dump_tokens.empty()) { FILE*f=fopen(dump_tokens.c_str(),"wb"); int64_t cc=C,tt=target_len; fwrite(&cc,8,1,f); fwrite(&tt,8,1,f); for(int i=0;i<C*target_len;i++){int64_t v=tokens[i];fwrite(&v,8,1,f);} fclose(f); }

    ggml_context * cwctx=nullptr; gguf_init_params cgp={false,&cwctx}; gguf_context* cg=gguf_init_from_file(codec.c_str(),cgp);
    if(!cg){ fprintf(stderr,"codec load fail\n"); return 1; }
    std::vector<int64_t> toks64((size_t)C*target_len); for(size_t i=0;i<toks64.size();i++) toks64[i]=tokens[i];
    auto tc0=std::chrono::high_resolution_clock::now();
    std::vector<float> wav=decode_codec(cwctx,toks64,C,target_len);  // CPU codec (f16 gguf)
    double codec_s=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tc0).count();
    double audio_s = wav.size()/24000.0;
    printf("codec: %.2fs | audio %.2fs | RTF %.2f (stage0+codec / audio)\n", codec_s, audio_s, (stage0_s+codec_s)/audio_s);

    FILE*f=fopen(out.c_str(),"wb"); uint32_t sr=24000,nb=wav.size()*2,chunk=36+nb; uint16_t one=1,bps=16,ba=2;
    fwrite("RIFF",1,4,f);fwrite(&chunk,4,1,f);fwrite("WAVE",1,4,f);fwrite("fmt ",1,4,f);
    uint32_t f16=16;fwrite(&f16,4,1,f);fwrite(&one,2,1,f);fwrite(&one,2,1,f);fwrite(&sr,4,1,f);
    uint32_t br=sr*2;fwrite(&br,4,1,f);fwrite(&ba,2,1,f);fwrite(&bps,2,1,f);fwrite("data",1,4,f);fwrite(&nb,4,1,f);
    for(float s:wav){int v=(int)lroundf(std::max(-1.0f,std::min(1.0f,s))*32767);int16_t pcm=(int16_t)v;fwrite(&pcm,2,1,f);} fclose(f);
    printf("wrote %s\n", out.c_str());

    ggml_gallocr_free(fc->galloc); ggml_free(fc->ctx); delete fc;
    ggml_gallocr_free(fu->galloc); ggml_free(fu->ctx); delete fu;
    gguf_free(gg); ggml_backend_buffer_free(gbuf); ggml_free(gwctx); ggml_backend_free(backend); gguf_free(cg); ggml_free(cwctx);
    return 0;
}
