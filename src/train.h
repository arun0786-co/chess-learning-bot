// self play -> learn weights -> new weights must beat the old ones -> repeat
#pragma once
#include <random>
#include <string>
#include <vector>

#include "eval.h"
#include "mcts.h"

namespace chess {

struct sample {
    float f[n_feat];
    float result;  // for the side to move: 1 win, 0.5 draw, 0 loss
};

struct game_stats {
    int games = 0, white_wins = 0, black_wins = 0, draws = 0;
    long plies = 0;
};

// plays games against itself and keeps every position with the final result
game_stats self_play(const weights& w, int games, const search_opts& o, int max_plies, std::mt19937& rng,
                     std::vector<sample>& out);

// logistic regression: find w so that sigmoid(w . f) is close to the real results
weights fit(const std::vector<sample>& data, const weights& start, int epochs, float l2, float* final_loss = nullptr);

// a vs b, both colours from a few openings. returns a's score from 0 to 1
float match(const weights& a, const weights& b, int games, const search_opts& o, int max_plies, std::mt19937& rng);

float elo_from_score(float s);

}  // namespace chess
