// monte carlo tree search. no random playouts: the leaf is judged by the learned eval.
#pragma once
#include <random>
#include <utility>
#include <vector>

#include "board.h"
#include "eval.h"

namespace chess {

struct search_opts {
    int playouts = 400;     // stop after this many (if movetime is 0)
    int movetime_ms = 0;    // or stop after this much time
    float c = 0.3f;         // how much to explore (small: trust the learned eval more)
    int quiet = 4;          // capture search depth at the leaves, 0 = off
    int batch = 1;          // leaves judged together (needs quiet = 0, used for the gpu)
};

struct search_result {
    move best;
    std::vector<std::pair<move, int>> visits;
    float value = 0.5f;  // chance to win for the side to move
    int playouts = 0;
};

// result of the game from here without searching: 1 / 0.5 / 0 for the side to move, or -1 if not over
float game_over(position& p, bool& over);

search_result mcts(position& p, const weights& w, const search_opts& o);

}  // namespace chess
