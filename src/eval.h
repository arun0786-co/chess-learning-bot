// how good is a position? a few simple features, and weights that the bot learns by itself.
// value = sigmoid(w . f) = chance that the side to move wins (draw = 0.5)
#pragma once
#include <string>

#include "board.h"

namespace chess {

constexpr int n_feat = 16;
extern const char* feat_names[n_feat];

struct weights {
    float w[n_feat] = {0};
    bool load(const std::string& path);
    bool save(const std::string& path) const;
};

// features from the side to move's point of view
void features(const position& p, float* f);
float logit(const weights& w, const position& p);
float sigmoid(float x);

// looks at captures until things are calm, so we dont judge a position in the middle of a trade
float quiet_logit(position& p, const weights& w, float alpha, float beta, int depth);

// many positions at once (cpu here, gpu version in cuda/mcts_kernel.cu)
void eval_batch(const float* feats, int n, const weights& w, float* out);

}  // namespace chess
