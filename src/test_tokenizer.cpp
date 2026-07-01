// P5 test: validate the C++ BPE tokenizer against known-good Qwen2 token ids.
#include "tokenizer.hpp"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <tokenizer.bin> [text]\n", argv[0]); return 1; }
    BpeTokenizer tk;
    if (!tk.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }

    struct Case { std::string text; std::vector<int> want; };
    std::vector<Case> cases = {
        {"Hello, this is a test of zero shot text to speech.",
         {9707,11,419,374,264,1273,315,7168,6552,1467,311,8806,13}},
        {"en", {268}},
        {"None", {4064}},
    };
    int fails = 0;
    for (auto & c : cases) {
        auto got = tk.encode(c.text);
        bool ok = got == c.want;
        printf("%-52s -> [", ("\"" + c.text + "\"").c_str());
        for (size_t i = 0; i < got.size(); i++) printf("%d%s", got[i], i + 1 < got.size() ? "," : "");
        printf("]  %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fails++; printf("   want: ["); for (int v : c.want) printf("%d,", v); printf("]\n"); }
    }
    if (argc > 2) {
        auto got = tk.encode(argv[2]);
        printf("\ncustom \"%s\" -> ", argv[2]);
        for (int v : got) printf("%d ", v);
        printf("\n");
    }
    printf("\nspecial ids: lang_start=%d lang_end=%d text_start=%d text_end=%d instruct_start=%d instruct_end=%d\n",
           tk.token_to_id("<|lang_start|>"), tk.token_to_id("<|lang_end|>"),
           tk.token_to_id("<|text_start|>"), tk.token_to_id("<|text_end|>"),
           tk.token_to_id("<|instruct_start|>"), tk.token_to_id("<|instruct_end|>"));
    printf("RESULT: %s\n", fails == 0 ? "PASS ✅" : "FAIL ❌");
    return fails ? 1 : 0;
}
