// Qwen2 byte-level BPE tokenizer for the OmniVoice C++ engine (P5).
// Loads models/tokenizer.bin (see tools/export_tokenizer.py). Reproduces the
// HF pipeline: GPT-2 regex pre-tokenize -> byte-level encode -> ranked BPE merges.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

class BpeTokenizer {
public:
    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[4]; uint32_t ver;
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVTK", 4) != 0) return false;
        if (fread(&ver, 4, 1, f) != 1) return false;
        auto ru32 = [&](uint32_t & v){ return fread(&v, 4, 1, f) == 1; };
        auto rstr = [&](std::string & s){ uint32_t n; if (!ru32(n)) return false; s.resize(n); return n == 0 || fread(&s[0], 1, n, f) == n; };
        uint32_t nv; ru32(nv);
        for (uint32_t i = 0; i < nv; i++) { uint32_t id; ru32(id); std::string t; rstr(t); vocab_[t] = (int) id; }
        uint32_t nm; ru32(nm);
        for (uint32_t i = 0; i < nm; i++) { std::string l, r; rstr(l); rstr(r); rank_[l + '\x01' + r] = (int) i; }
        uint32_t na; ru32(na);
        for (uint32_t i = 0; i < na; i++) { uint32_t id; ru32(id); std::string t; rstr(t); vocab_[t] = (int) id; added_[t] = (int) id; }
        fclose(f);
        build_byte_map();
        return true;
    }

    int token_to_id(const std::string & tok) const {
        auto it = vocab_.find(tok); return it == vocab_.end() ? -1 : it->second;
    }

    std::vector<int> encode(const std::string & text) const {
        std::vector<int> out;
        for (auto & piece : pretokenize(text)) {
            // byte-level encode each raw byte of the piece into a symbol string
            std::vector<std::string> syms;
            for (unsigned char b : piece) syms.push_back(byte2str_[b]);
            bpe_merge(syms);
            for (auto & s : syms) {
                auto it = vocab_.find(s);
                if (it != vocab_.end()) out.push_back(it->second);
                // (byte-level guarantees single-byte symbols are always in vocab)
            }
        }
        return out;
    }

private:
    std::unordered_map<std::string, int> vocab_, added_, rank_;
    std::string byte2str_[256];

    void build_byte_map() {
        // GPT-2 byte<->unicode: printable ranges map to themselves, others to 256+n.
        std::vector<int> cp(256);
        std::vector<bool> used(256, false);
        auto add_range = [&](int lo, int hi){ for (int b = lo; b <= hi; b++) { cp[b] = b; used[b] = true; } };
        add_range('!', '~'); add_range(0xA1, 0xAC); add_range(0xAE, 0xFF);
        int n = 0;
        for (int b = 0; b < 256; b++) if (!used[b]) { cp[b] = 256 + n; n++; }
        for (int b = 0; b < 256; b++) byte2str_[b] = utf8(cp[b]);
    }
    static std::string utf8(int c) {
        std::string s;
        if (c < 0x80) s += (char) c;
        else if (c < 0x800) { s += (char) (0xC0 | (c >> 6)); s += (char) (0x80 | (c & 0x3F)); }
        else { s += (char) (0xE0 | (c >> 12)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F)); }
        return s;
    }

    void bpe_merge(std::vector<std::string> & syms) const {
        while (syms.size() > 1) {
            int best = INT32_MAX, bi = -1;
            for (size_t i = 0; i + 1 < syms.size(); i++) {
                auto it = rank_.find(syms[i] + '\x01' + syms[i + 1]);
                if (it != rank_.end() && it->second < best) { best = it->second; bi = (int) i; }
            }
            if (bi < 0) break;
            syms[bi] += syms[bi + 1];
            syms.erase(syms.begin() + bi + 1);
        }
    }

    // --- UTF-8 pre-tokenizer approximating the GPT-2/Qwen regex ---
    struct CP { int cp; int lo, len; }; // codepoint + byte span
    static std::vector<CP> decode_utf8(const std::string & t) {
        std::vector<CP> v; size_t i = 0;
        while (i < t.size()) {
            unsigned char c = t[i]; int cp, len;
            if (c < 0x80) { cp = c; len = 1; }
            else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
            else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
            else { cp = c & 0x07; len = 4; }
            for (int k = 1; k < len && i + k < t.size(); k++) cp = (cp << 6) | (t[i + k] & 0x3F);
            v.push_back({cp, (int) i, len}); i += len;
        }
        return v;
    }
    static bool is_ws(int c){ return c==' '||c=='\t'||c=='\r'||c=='\n'||c==0x0B||c==0x0C||c==0xA0; }
    static bool is_letter(int c){ return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>=0x80 && !is_ws(c)); }
    static bool is_num(int c){ return c>='0'&&c<='9'; }
    static bool is_nl(int c){ return c=='\r'||c=='\n'; }

    std::vector<std::string> pretokenize(const std::string & text) const {
        auto cps = decode_utf8(text);
        std::vector<std::string> pieces;
        size_t i = 0, N = cps.size();
        auto span = [&](size_t a, size_t b){ return text.substr(cps[a].lo, cps[b - 1].lo + cps[b - 1].len - cps[a].lo); };
        auto low = [](int c){ return (c>='A'&&c<='Z') ? c+32 : c; };
        while (i < N) {
            int c = cps[i].cp;
            // 1. contractions 's 't 're 've 'm 'll 'd (case-insensitive)
            if (c == '\'' && i + 1 < N) {
                int a = low(cps[i+1].cp);
                int b = (i + 2 < N) ? low(cps[i+2].cp) : 0;
                size_t adv = 0;
                if (a=='s'||a=='t'||a=='m'||a=='d') adv = 2;
                else if ((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) adv = 3;
                if (adv) { pieces.push_back(span(i, i+adv)); i += adv; continue; }
            }
            // 2. [^\r\n\p{L}\p{N}]? \p{L}+   (optional leading non-letter/num, then letters)
            {
                size_t j = i;
                if (!is_nl(c) && !is_letter(c) && !is_num(c) && (i+1<N) && is_letter(cps[i+1].cp)) j = i + 1; // optional leading
                if (j < N && is_letter(cps[j].cp)) {
                    size_t k = j; while (k < N && is_letter(cps[k].cp)) k++;
                    pieces.push_back(span(i, k)); i = k; continue;
                }
            }
            // 3. single number
            if (is_num(c)) { pieces.push_back(span(i, i+1)); i++; continue; }
            // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*   (optional space, punct run, trailing newlines)
            {
                size_t j = i;
                if (c == ' ' && (i+1<N) && !is_ws(cps[i+1].cp) && !is_letter(cps[i+1].cp) && !is_num(cps[i+1].cp)) j = i + 1;
                if (j < N && !is_ws(cps[j].cp) && !is_letter(cps[j].cp) && !is_num(cps[j].cp)) {
                    size_t k = j; while (k < N && !is_ws(cps[k].cp) && !is_letter(cps[k].cp) && !is_num(cps[k].cp)) k++;
                    while (k < N && is_nl(cps[k].cp)) k++;
                    pieces.push_back(span(i, k)); i = k; continue;
                }
            }
            // 5-7. whitespace runs (\s+, keeping a trailing space attached to next word is
            // already handled by rule 2; here we just emit remaining whitespace runs)
            if (is_ws(c)) {
                size_t k = i; while (k < N && is_ws(cps[k].cp)) k++;
                // \s+(?!\S): if not at end, leave the last space for the next word (rule 2)
                if (k < N && k - i > 1) k--;
                pieces.push_back(span(i, k)); i = k; continue;
            }
            // fallback: single codepoint
            pieces.push_back(span(i, i+1)); i++;
        }
        return pieces;
    }
};
