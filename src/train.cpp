#include "train.h"

#include <algorithm>
#include <cmath>

namespace chess {

// pick a move from the visit counts. early in the game a bit random so games are different
static move choose(const search_result& r, bool explore, std::mt19937& rng) {
    if (!explore || r.visits.empty()) return r.best;
    int total = 0;
    for (auto& v : r.visits) total += v.second + 1;
    int x = std::uniform_int_distribution<int>(0, total - 1)(rng);
    for (auto& v : r.visits) {
        x -= v.second + 1;
        if (x < 0) return v.first;
    }
    return r.best;
}

static float play_one(const weights& wa, const weights& wb, const search_opts& o, int max_plies, int random_plies,
                      std::mt19937& rng, std::vector<sample>* out, long* plies) {
    // wa plays white. returns result for white
    position p;
    std::vector<sample> game;
    std::vector<int> sides;
    float white_result = 0.5f;
    for (int ply = 0; ply < max_plies; ply++) {
        bool over;
        float r = game_over(p, over);
        if (over) {
            white_result = p.side == white ? r : 1.0f - r;
            break;
        }
        if (out) {
            sample s;
            features(p, s.f);
            game.push_back(s);
            sides.push_back(p.side);
        }
        const weights& w = p.side == white ? wa : wb;
        search_result res = mcts(p, w, o);
        p.make(choose(res, ply < random_plies, rng));
        if (plies) (*plies)++;
    }
    if (out) {
        for (size_t i = 0; i < game.size(); i++) {
            game[i].result = sides[i] == white ? white_result : 1.0f - white_result;
            out->push_back(game[i]);
        }
    }
    return white_result;
}

game_stats self_play(const weights& w, int games, const search_opts& o, int max_plies, std::mt19937& rng,
                     std::vector<sample>& out) {
    game_stats st;
    for (int g = 0; g < games; g++) {
        float r = play_one(w, w, o, max_plies, 12, rng, &out, &st.plies);
        st.games++;
        if (r == 1.0f) st.white_wins++;
        else if (r == 0.0f) st.black_wins++;
        else st.draws++;
    }
    return st;
}

weights fit(const std::vector<sample>& data, const weights& start, int epochs, float l2, float* final_loss) {
    weights w = start;
    float m[n_feat] = {0}, v[n_feat] = {0};
    const float lr = 0.05f;
    float loss = 0;
    for (int ep = 1; ep <= epochs; ep++) {
        float g[n_feat] = {0};
        loss = 0;
        for (const sample& s : data) {
            float z = 0;
            for (int i = 0; i < n_feat; i++) z += w.w[i] * s.f[i];
            float p = sigmoid(z);
            float pc = std::min(std::max(p, 1e-6f), 1 - 1e-6f);
            loss += -(s.result * std::log(pc) + (1 - s.result) * std::log(1 - pc));
            for (int i = 0; i < n_feat; i++) g[i] += (p - s.result) * s.f[i];
        }
        float n = (float)std::max<size_t>(1, data.size());
        loss /= n;
        for (int i = 0; i < n_feat; i++) {
            float gi = g[i] / n + l2 * w.w[i];
            m[i] = 0.9f * m[i] + 0.1f * gi;
            v[i] = 0.999f * v[i] + 0.001f * gi * gi;
            float mh = m[i] / (1 - std::pow(0.9f, (float)ep));
            float vh = v[i] / (1 - std::pow(0.999f, (float)ep));
            w.w[i] -= lr * mh / (std::sqrt(vh) + 1e-8f);
        }
    }
    if (final_loss) *final_loss = loss;
    return w;
}

float match(const weights& a, const weights& b, int games, const search_opts& o, int max_plies, std::mt19937& rng) {
    float score = 0;
    for (int g = 0; g < games; g++) {
        // same randomness for both colours, so the openings are fair
        if (g % 2 == 0) score += play_one(a, b, o, max_plies, 6, rng, nullptr, nullptr);
        else score += 1.0f - play_one(b, a, o, max_plies, 6, rng, nullptr, nullptr);
    }
    return games ? score / games : 0.5f;
}

float elo_from_score(float s) {
    s = std::min(std::max(s, 0.01f), 0.99f);
    return -400.0f * std::log10(1.0f / s - 1.0f);
}

}  // namespace chess
