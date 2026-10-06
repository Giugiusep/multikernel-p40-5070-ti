#include "llama.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static llama_token run(std::vector<llama_token_data> cur, bool full) {
    llama_token_data_array a{cur.data(), cur.size(), -1, false};
    std::vector<llama_sampler *> ss;
    if (full) {
        ss = {
            llama_sampler_init_top_n_sigma(-1.0f),
            llama_sampler_init_top_k(40),
            llama_sampler_init_typical(1.0f, 0),
            llama_sampler_init_top_p(0.95f, 0),
            llama_sampler_init_min_p(0.05f, 0),
            llama_sampler_init_xtc(0.0f, 0.10f, 0, 42),
        };
    }
    ss.push_back(llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    ss.push_back(llama_sampler_init_dist(42));
    for (auto * s : ss) llama_sampler_apply(s, &a);
    for (auto * s : ss) llama_sampler_free(s);
    return a.data[a.selected].id;
}

static llama_token run_greedy(std::vector<llama_token_data> cur) {
    llama_token_data_array a{cur.data(), cur.size(), -1, false};
    llama_sampler * sampler = llama_sampler_init_greedy();
    llama_sampler_apply(sampler, &a);
    llama_sampler_free(sampler);
    return a.data[a.selected].id;
}

int main() {
    constexpr int vocab = 131072;
    constexpr int trials = 100;
    std::mt19937 rng(0x5a17);
    std::uniform_real_distribution<float> d(-30.0f, 30.0f);
    for (int t = 0; t < trials; ++t) {
        std::vector<llama_token_data> x;
        x.reserve(vocab);
        for (int i = 0; i < vocab; ++i) x.push_back({i, d(rng) + i*1e-8f, 0.0f});
        auto expected = std::max_element(x.begin(), x.end(), [](auto &a, auto &b){return a.logit < b.logit;})->id;
        auto full = run(x, true), minimal = run(x, false), greedy = run_greedy(x);
        if (full != expected || minimal != expected || greedy != expected) {
            std::fprintf(stderr, "trial=%d expected=%d full=%d minimal=%d greedy=%d\n",
                    t, expected, full, minimal, greedy);
            return 1;
        }
    }
    std::printf("PASS trials=%d vocab=%d\n", trials, vocab);
}
