// small checks for the rules helpers, the eval, the search and the learning
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "../src/board.h"
#include "../src/eval.h"
#include "../src/mcts.h"
#include "../src/train.h"

using namespace chess;

static int failed = 0;
#define check(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);        \
            failed++;                                                  \
        }                                                              \
    } while (0)

static void play(position& p, const char* moves) {
    std::string s = moves, tok;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ' ') {
            move m;
            if (!tok.empty() && p.parse_move(tok, m)) p.make(m);
            tok.clear();
        } else {
            tok += s[i];
        }
    }
}

int main() {
    // fen goes in and comes out the same
    position p;
    const char* kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    p.set_fen(kiwi);
    check(p.fen() == kiwi);
    check(p.hash == p.compute_hash());

    // make + unmake leaves everything as it was
    p.set_fen(start_fen);
    play(p, "e2e4 d7d5 e4d5 g8f6 f1b5 c7c6 d5c6 d8a5 c6b7 a5b5 b7a8q");
    check(p.hash == p.compute_hash());
    for (int i = 0; i < 11; i++) p.unmake();
    check(p.fen() == start_fen);

    // knights going back and forth = repetition
    p.set_fen(start_fen);
    play(p, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
    check(p.repetitions() == 2);
    bool over;
    check(game_over(p, over) == 0.5f && over);

    // stalemate and mate
    p.set_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    check(game_over(p, over) == 0.5f && over);
    p.set_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    check(game_over(p, over) == 0.0f && over);
    p.set_fen("8/8/4k3/8/8/2K5/8/8 w - - 0 1");
    check(p.insufficient());

    // features: start position is equal, and seen the same from both sides
    float f[n_feat], g[n_feat];
    p.set_fen(start_fen);
    features(p, f);
    for (int i = 0; i < 14; i++) check(f[i] == 0.0f);
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1");  // white without a knight
    features(p, f);
    check(f[1] == -1.0f);
    p.set_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");  // same thing, colours swapped
    features(p, g);
    for (int i = 0; i < n_feat; i++) check(std::fabs(f[i] - g[i]) < 1e-6f);

    // the batch eval gives the same as one by one
    weights w;
    for (int i = 0; i < n_feat; i++) w.w[i] = 0.1f * (i + 1);
    p.set_fen(kiwi);
    features(p, f);
    float out;
    eval_batch(f, 1, w, &out);
    check(std::fabs(out - sigmoid(logit(w, p))) < 1e-5f);

    // search finds mate in one even with zero weights, and grabs a free queen with material weights
    weights zero;
    search_opts o;
    o.playouts = 300;
    p.set_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
    check(p.move_str(mcts(p, zero, o).best) == "a1a8");
    weights mat;
    mat.w[0] = 1, mat.w[1] = 3, mat.w[2] = 3, mat.w[3] = 5, mat.w[4] = 9;
    for (int i = 0; i < 5; i++) mat.w[i] *= 0.3f;
    p.set_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1");
    check(p.move_str(mcts(p, mat, o).best) == "d1d5");

    // learning: fake data where only feature 0 matters -> its weight comes out big and positive
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    std::vector<sample> data;
    for (int i = 0; i < 4000; i++) {
        sample s{};
        s.f[0] = nd(rng);
        s.f[1] = nd(rng);
        s.f[15] = 1;
        s.result = std::uniform_real_distribution<float>(0, 1)(rng) < sigmoid(2.0f * s.f[0]) ? 1.0f : 0.0f;
        data.push_back(s);
    }
    weights fitted = fit(data, weights(), 400, 0.0f);
    check(std::fabs(fitted.w[0] - 2.0f) < 0.3f);
    check(std::fabs(fitted.w[1]) < 0.2f);

    check(std::fabs(elo_from_score(0.5f)) < 1e-3f);
    check(elo_from_score(0.75f) > 180 && elo_from_score(0.75f) < 200);

    std::printf(failed ? "%d checks failed\n" : "all good\n", failed);
    return failed ? 1 : 0;
}
