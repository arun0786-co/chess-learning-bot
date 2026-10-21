#include "mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace chess {

struct node {
    move m;
    int parent = -1;
    int first = -1;   // first child in the node list
    int count = 0;    // number of children
    int visits = 0;
    float total = 0;  // summed value, seen by the player who made move m
    bool expanded = false;
    bool terminal = false;
    float end_value = 0;  // for terminal nodes, also seen by the player who made m
};

float game_over(position& p, bool& over) {
    over = true;
    if (p.halfmove >= 100 || p.insufficient() || p.repetitions() >= 2) return 0.5f;
    std::vector<move> legal;
    p.gen_legal(legal);
    if (legal.empty()) return p.in_check() ? 0.0f : 0.5f;
    over = false;
    return -1;
}

static int victim_score(const position& p, const move& m) {
    static const int v[7] = {0, 1, 3, 3, 5, 9, 0};
    int t = p.sq[m.to] > 0 ? p.sq[m.to] : -p.sq[m.to];
    return v[t] * 10 + (m.promo ? 50 : 0);
}

static void expand(std::vector<node>& tree, int id, position& p) {
    std::vector<move> legal;
    p.gen_legal(legal);
    // captures and promotions first, they get tried first while the node has no visits
    std::stable_sort(legal.begin(), legal.end(),
                     [&](const move& a, const move& b) { return victim_score(p, a) > victim_score(p, b); });
    tree[id].expanded = true;
    tree[id].first = (int)tree.size();
    tree[id].count = (int)legal.size();
    for (const move& m : legal) {
        node c;
        c.m = m;
        c.parent = id;
        tree.push_back(c);
    }
}

static int pick_child(const std::vector<node>& tree, int id, float c) {
    const node& n = tree[id];
    float log_n = std::log((float)std::max(1, n.visits));
    int best = -1;
    float best_score = -1e9f;
    for (int i = n.first; i < n.first + n.count; i++) {
        const node& ch = tree[i];
        if (ch.visits == 0) return i;
        float q = ch.total / ch.visits;
        float score = q + c * std::sqrt(log_n / ch.visits);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

search_result mcts(position& p, const weights& w, const search_opts& o) {
    std::vector<node> tree;
    tree.reserve(1 << 16);
    tree.push_back(node());
    expand(tree, 0, p);
    search_result res;
    if (tree[0].count == 0) return res;

    auto t0 = std::chrono::steady_clock::now();
    auto time_up = [&]() {
        if (o.movetime_ms <= 0) return res.playouts >= o.playouts;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        return ms >= o.movetime_ms;
    };
    int batch = o.quiet > 0 ? 1 : std::max(1, o.batch);
    std::vector<float> feats(batch * n_feat), vals(batch);
    std::vector<std::vector<int>> paths(batch);
    std::vector<float> leaf_value(batch);
    std::vector<bool> need_eval(batch);

    while (!time_up() && tree[0].count > 1) {
        int got = 0;
        for (int b = 0; b < batch; b++) {
            auto& path = paths[b];
            path.assign(1, 0);
            int id = 0;
            int depth = 0;
            while (tree[id].expanded && !tree[id].terminal && tree[id].count > 0) {
                id = pick_child(tree, id, o.c);
                p.make(tree[id].m);
                path.push_back(id);
                depth++;
            }
            node& leaf = tree[id];
            need_eval[b] = false;
            if (leaf.terminal) {
                leaf_value[b] = leaf.end_value;
            } else {
                bool over;
                float r = game_over(p, over);
                if (over) {
                    leaf.terminal = true;
                    leaf.end_value = 1.0f - r;  // r is for the side to move, we store it for the mover
                    leaf_value[b] = leaf.end_value;
                } else {
                    expand(tree, id, p);
                    if (o.quiet > 0) {
                        float v = sigmoid(quiet_logit(p, w, -1e9f, 1e9f, o.quiet));
                        leaf_value[b] = 1.0f - v;
                    } else {
                        features(p, &feats[got * n_feat]);
                        need_eval[b] = true;
                    }
                }
            }
            // virtual loss: count the visit now so the next leaf in this batch goes somewhere else
            for (int x : path) tree[x].visits++;
            for (int i = 0; i < depth; i++) p.unmake();
            if (need_eval[b]) got++;
        }
        if (got) eval_batch(feats.data(), got, w, vals.data());
        int k = 0;
        for (int b = 0; b < batch; b++) {
            float v = need_eval[b] ? 1.0f - vals[k++] : leaf_value[b];
            auto& path = paths[b];
            for (int i = (int)path.size() - 1; i >= 0; i--) {
                tree[path[i]].total += v;
                v = 1.0f - v;
            }
        }
        res.playouts += batch;
    }

    int best = tree[0].first;
    for (int i = tree[0].first; i < tree[0].first + tree[0].count; i++) {
        res.visits.push_back({tree[i].m, tree[i].visits});
        if (tree[i].visits > tree[best].visits) best = i;
    }
    res.best = tree[best].m;
    res.value = tree[best].visits ? tree[best].total / tree[best].visits : 0.5f;
    return res;
}

}  // namespace chess
