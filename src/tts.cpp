// P5/P6: end-to-end OmniVoice TTS in C++/ggml.  text -> 24kHz WAV.
// Pipeline: BPE tokenize -> build prompt -> Stage0 masked diffusion (generator)
//           -> RVQ+DAC codec -> WAV.  All neural stages on ggml (CPU).
//
// Usage: tts <generator.gguf> <codec.gguf> <tokenizer.bin> <out.wav>
//        --text "..." [--lang en] [--instruct None] [--duration SEC]
//        [--num-step 32] [--guidance 2.0] [--t-shift 0.1] [--layer-penalty 5.0]
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static ggml_tensor * must(ggml_context * c, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(c, n.c_str());
    if (!t) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
    return t;
}
static uint32_t kv_u32(gguf_context * g, const char * k, uint32_t d) { int64_t i = gguf_find_key(g, k); return i < 0 ? d : gguf_get_val_u32(g, i); }
static float   kv_f32(gguf_context * g, const char * k, float d)    { int64_t i = gguf_find_key(g, k); return i < 0 ? d : gguf_get_val_f32(g, i); }

struct Hparams { int n_layer, n_embd, n_head, n_kv, hd; float eps, theta; };

// ---- Qwen3 backbone + audio heads on a prebuilt embedding stream. Returns [1025,8,S]. ----
static std::vector<float> forward_audio_logits(ggml_context * wctx, const Hparams & hp, int n_cb, int vocab,
        const std::vector<int32_t> & text_ids, const std::vector<int32_t> & aud_ids, const std::vector<float> & amask, int S) {
    ggml_init_params cp = { (size_t) 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * tids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    memcpy(tids->data, text_ids.data(), S * sizeof(int32_t));
    ggml_tensor * temb = ggml_get_rows(ctx, must(wctx, "token_embd.weight"), tids);
    if (temb->type != GGML_TYPE_F32) temb = ggml_cast(ctx, temb, GGML_TYPE_F32);
    ggml_tensor * aemb_w = must(wctx, "audio_embeddings.weight");
    ggml_tensor * aud_sum = nullptr;
    for (int c = 0; c < n_cb; c++) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        int32_t * p = (int32_t *) ids->data;
        for (int t = 0; t < S; t++) p[t] = aud_ids[(size_t) c * S + t];
        ggml_tensor * e = ggml_get_rows(ctx, aemb_w, ids);
        if (e->type != GGML_TYPE_F32) e = ggml_cast(ctx, e, GGML_TYPE_F32);
        aud_sum = aud_sum ? ggml_add(ctx, aud_sum, e) : e;
    }
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S);
    ggml_tensor * mm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, S);
    for (int t = 0; t < S; t++) { ((float*)m->data)[t] = amask[t]; ((float*)mm->data)[t] = 1.0f - amask[t]; }
    ggml_tensor * cur = ggml_add(ctx, ggml_mul(ctx, aud_sum, m), ggml_mul(ctx, temb, mm));
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
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), must(wctx, "output_norm.weight"));
    ggml_tensor * logits = ggml_mul_mat(ctx, must(wctx, "audio_heads.weight"), cur);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(ctx, gf, 8);
    std::vector<float> out((size_t) vocab * n_cb * S);
    memcpy(out.data(), logits->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// ---- DAC codec decode (see test_codec.cpp): tokens[C,T] -> waveform ----
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

// ---- pragmatic duration estimator (ports frontend/duration.rs char weights) ----
static float char_weight(int cp) {
    if ((cp>='A'&&cp<='Z')||(cp>='a'&&cp<='z')) return 1.0f;   // latin
    if (cp==' ') return 0.2f;
    if (cp>='0'&&cp<='9') return 3.5f;                          // ascii digit
    if (cp<0x80) return 0.5f;                                   // ascii punct/symbol
    // Devanagari & Indic combining marks -> 0 (matras); base letters -> 1.8
    if ((cp>=0x0900&&cp<=0x0903)||(cp>=0x093A&&cp<=0x094F)||(cp>=0x0951&&cp<=0x0957)||(cp>=0x0962&&cp<=0x0963)) return 0.0f;
    if (cp>=0x0900&&cp<=0x0DFF) return 1.8f;                    // indic scripts
    if (cp>=0x0E00&&cp<=0x0EFF) return 1.5f;                    // thai/lao
    if (cp>=0x4E00&&cp<=0x9FFF) return 3.0f;                    // CJK
    if (cp>=0x0400&&cp<=0x04FF) return 1.0f;                    // cyrillic
    if (cp>=0x0600&&cp<=0x06FF) return 1.5f;                    // arabic
    return 1.0f;
}
static std::vector<int> utf8_cps(const std::string & t) {
    std::vector<int> v; size_t i = 0;
    while (i < t.size()) { unsigned char c = t[i]; int cp, len;
        if (c<0x80){cp=c;len=1;} else if((c>>5)==0x6){cp=c&0x1F;len=2;} else if((c>>4)==0xE){cp=c&0x0F;len=3;} else {cp=c&0x07;len=4;}
        for (int k=1;k<len&&i+k<t.size();k++) cp=(cp<<6)|(t[i+k]&0x3F);
        v.push_back(cp); i+=len; } return v;
}
static float total_weight(const std::string & t) { float s = 0; for (int cp : utf8_cps(t)) s += char_weight(cp); return s; }
static int estimate_target_len(const std::string & text) {
    const char * ref = "Nice to meet you."; float ref_dur = 25.0f, low = 50.0f, alpha = 1.0f/3.0f;
    float rw = total_weight(ref); if (rw == 0) return 25;
    float est = total_weight(text) / (rw / ref_dur);
    if (est < low) est = low * powf(est / low, alpha);
    return std::max(1, (int) est);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <gen.gguf> <codec.gguf> <tok.bin> <out.wav> --text \"...\" [--lang en] [--instruct None] [--duration S] [--num-step N] [--guidance G] [--t-shift T] [--layer-penalty L]\n", argv[0]); return 1; }
    std::string gen = argv[1], codec = argv[2], tokpath = argv[3], out = argv[4];
    std::string text, lang = "en", instruct = "None"; float duration = -1, guidance = 2.0f, t_shift = 0.1f, lpf = 5.0f; int num_step = 32;
    for (int i = 5; i < argc; i++) { std::string a = argv[i];
        auto nx = [&]{ return (i+1<argc) ? argv[++i] : ""; };
        if (a=="--text") text = nx(); else if (a=="--lang") lang = nx(); else if (a=="--instruct") instruct = nx();
        else if (a=="--duration") duration = atof(nx()); else if (a=="--num-step") num_step = atoi(nx());
        else if (a=="--guidance") guidance = atof(nx()); else if (a=="--t-shift") t_shift = atof(nx());
        else if (a=="--layer-penalty") lpf = atof(nx()); }
    if (text.empty()) { fprintf(stderr, "--text required\n"); return 1; }

    BpeTokenizer tk; if (!tk.load(tokpath)) { fprintf(stderr, "tokenizer load fail\n"); return 1; }
    // build prompt: <lang_start> lang <lang_end> <instruct_start> instruct <instruct_end> <text_start> text <text_end>
    std::vector<int> prompt;
    auto add = [&](const std::string & sp){ prompt.push_back(tk.token_to_id(sp)); };
    add("<|lang_start|>"); for (int id : tk.encode(lang)) prompt.push_back(id); add("<|lang_end|>");
    add("<|instruct_start|>"); for (int id : tk.encode(instruct)) prompt.push_back(id); add("<|instruct_end|>");
    add("<|text_start|>"); for (int id : tk.encode(text)) prompt.push_back(id); add("<|text_end|>");
    int text_len = (int) prompt.size();
    int target_len = (duration > 0) ? std::max(1, (int) lroundf(duration * 25.0f)) : estimate_target_len(text);
    int cond_len = text_len + target_len;
    printf("text_tokens=%d target_len=%d (%.2fs) cond_len=%d num_step=%d\n", text_len, target_len, target_len/25.0, cond_len, num_step);

    // load generator
    ggml_context * gwctx = nullptr; gguf_init_params gp = { false, &gwctx };
    gguf_context * gg = gguf_init_from_file(gen.c_str(), gp);
    if (!gg) { fprintf(stderr, "gen load fail\n"); return 1; }
    Hparams hp { (int)kv_u32(gg,"qwen3.block_count",28),(int)kv_u32(gg,"qwen3.embedding_length",1024),
        (int)kv_u32(gg,"qwen3.attention.head_count",16),(int)kv_u32(gg,"qwen3.attention.head_count_kv",8),
        (int)kv_u32(gg,"qwen3.attention.key_length",128),kv_f32(gg,"qwen3.attention.layer_norm_rms_epsilon",1e-6f),kv_f32(gg,"qwen3.rope.freq_base",1e6f) };
    const int C = (int)kv_u32(gg,"omnivoice.num_audio_codebooks",8), vocab = (int)kv_u32(gg,"omnivoice.audio_vocab_size",1025), mask_id = (int)kv_u32(gg,"omnivoice.audio_mask_id",1024);
    std::vector<int> offset(C); for (int c=0;c<C;c++) offset[c]=c*vocab;

    // build cond/uncond id grids + audio masks
    int L = cond_len;
    std::vector<int32_t> ids0((size_t)C*L, 0), ids1((size_t)C*target_len, mask_id);
    for (int t=0;t<text_len;t++) ids0[t] = prompt[t];                 // cb0 text region
    for (int c=0;c<C;c++) for (int t=text_len;t<L;t++) ids0[c*L+t] = mask_id;  // audio region masked
    std::vector<float> am0(L,0.0f), am1(target_len,1.0f);
    for (int t=text_len;t<L;t++) am0[t]=1.0f;

    std::vector<float> ts(num_step+2); for (int i=0;i<=num_step+1;i++){ float t=(float)i/(num_step+1); ts[i]=(t_shift*t)/(1+(t_shift-1)*t); }
    int total=target_len*C, remaining=total; std::vector<int> sched(num_step);
    for (int s=0;s<num_step;s++){ int amt=(s==num_step-1)?remaining:(int)std::ceil(total*(ts[s+1]-ts[s])); amt=std::min(amt,remaining); sched[s]=amt; remaining-=amt; }

    std::vector<int32_t> tokens((size_t)C*target_len, mask_id);
    auto text_ids_of=[&](std::vector<int32_t>&ids,int S){ std::vector<int32_t> t(S); for(int i=0;i<S;i++) t[i]=ids[i]; return t; };
    auto aud_ids_of=[&](std::vector<int32_t>&ids,std::vector<float>&am,int S,int rowlen){ std::vector<int32_t> a((size_t)C*S);
        for(int c=0;c<C;c++)for(int t=0;t<S;t++){ int32_t raw=(am[t]>0.5f)?ids[c*rowlen+t]:0; a[c*S+t]=raw+offset[c]; } return a; };

    for (int step=0; step<num_step; step++) {
        if (sched[step]==0) continue;
        auto cl=forward_audio_logits(gwctx,hp,C,vocab,text_ids_of(ids0,cond_len),aud_ids_of(ids0,am0,cond_len,L),am0,cond_len);
        auto ul=forward_audio_logits(gwctx,hp,C,vocab,text_ids_of(ids1,target_len),aud_ids_of(ids1,am1,target_len,target_len),am1,target_len);
        int aud_start=cond_len-target_len;
        std::vector<int32_t> pred((size_t)C*target_len); std::vector<float> conf((size_t)C*target_len);
        for (int c=0;c<C;c++) for (int t=0;t<target_len;t++) {
            const float* cv=&cl[((size_t)(aud_start+t)*C+c)*vocab]; const float* uv=&ul[((size_t)t*C+c)*vocab];
            auto lse=[&](const float*z){ float mx=-1e30f; for(int v=0;v<vocab;v++) mx=std::max(mx,z[v]); double s=0; for(int v=0;v<vocab;v++) s+=exp(z[v]-mx); return mx+(float)log(s); };
            float lcz=lse(cv), luz=lse(uv); std::vector<float> guided(vocab); float gmx=-1e30f;
            for (int v=0;v<vocab;v++){ float lc=cv[v]-lcz, lu=uv[v]-luz; guided[v]=(1.0f+guidance)*lc-guidance*lu; gmx=std::max(gmx,guided[v]); }
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
    printf("stage0 done: %d tokens\n", C*target_len);

    // codec decode
    ggml_context * cwctx = nullptr; gguf_init_params cgp = { false, &cwctx };
    gguf_context * cg = gguf_init_from_file(codec.c_str(), cgp);
    if (!cg) { fprintf(stderr, "codec load fail\n"); return 1; }
    std::vector<int64_t> toks64((size_t)C*target_len); for (size_t i=0;i<toks64.size();i++) toks64[i]=tokens[i];
    std::vector<float> wav = decode_codec(cwctx, toks64, C, target_len);
    printf("decoded %zu samples (%.2fs)\n", wav.size(), wav.size()/24000.0);

    // write WAV (16-bit PCM mono 24kHz)
    FILE * f = fopen(out.c_str(), "wb");
    uint32_t sr=24000, nb=wav.size()*2, chunk=36+nb; uint16_t one=1, bps=16, ba=2;
    fwrite("RIFF",1,4,f); fwrite(&chunk,4,1,f); fwrite("WAVE",1,4,f); fwrite("fmt ",1,4,f);
    uint32_t f16=16; fwrite(&f16,4,1,f); fwrite(&one,2,1,f); fwrite(&one,2,1,f); fwrite(&sr,4,1,f);
    uint32_t br=sr*2; fwrite(&br,4,1,f); fwrite(&ba,2,1,f); fwrite(&bps,2,1,f); fwrite("data",1,4,f); fwrite(&nb,4,1,f);
    for (float s : wav) { int v=(int)lroundf(std::max(-1.0f,std::min(1.0f,s))*32767); int16_t pcm=(int16_t)v; fwrite(&pcm,2,1,f); }
    fclose(f);
    printf("wrote %s\n", out.c_str());
    gguf_free(gg); ggml_free(gwctx); gguf_free(cg); ggml_free(cwctx);
    return 0;
}
