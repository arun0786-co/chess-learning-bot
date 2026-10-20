#include "eval.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace chess {

const char* feat_names[n_feat] = {
    "pawn", "knight", "bishop", "rook", "queen", "bishop_pair", "mobility", "pawn_advance",
    "passed_pawn", "doubled_pawn", "isolated_pawn", "center", "king_shelter", "rook_open_file",
    "in_check", "tempo",
};

bool weights::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    std::string name;
    for (int i = 0; i < n_feat; i++) {
        if (!(in >> name >> w[i])) return false;
    }
    return true;
}

bool weights::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    for (int i = 0; i < n_feat; i++) out << feat_names[i] << " " << w[i] << "\n";
    return true;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static int mobility(const position& p, int color) {
    // pseudo moves of knights, bishops, rooks, queens. cheap and good enough
    static const int knight_d[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    static const int line_d[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    int n = 0, sign = color == white ? 1 : -1;
    for (int s = 0; s < 64; s++) {
        int t = p.sq[s] * sign;
        if (t < knight || t > queen) continue;
        int r = s / 8, f = s % 8;
        if (t == knight) {
            for (auto& d : knight_d) {
                int rr = r + d[0], ff = f + d[1];
                if (rr < 0 || rr > 7 || ff < 0 || ff > 7) continue;
                int8_t q = p.sq[rr * 8 + ff];
                if (q * sign <= 0) n++;
            }
            continue;
        }
        int from = t == rook ? 0 : (t == bishop ? 4 : 0);
        int to = t == bishop ? 8 : (t == rook ? 4 : 8);
        for (int i = from; i < to; i++) {
            int rr = r + line_d[i][0], ff = f + line_d[i][1];
            while (rr >= 0 && rr <= 7 && ff >= 0 && ff <= 7) {
                int8_t q = p.sq[rr * 8 + ff];
                if (q * sign > 0) break;
                n++;
                if (q) break;
                rr += line_d[i][0];
                ff += line_d[i][1];
            }
        }
    }
    return n;
}

void features(const position& p, float* f) {
    float x[n_feat] = {0};  // white minus black first
    int pawns_on_file[2][8] = {{0}};
    int count[2][7] = {{0}};
    for (int s = 0; s < 64; s++) {
        int8_t pc = p.sq[s];
        if (!pc) continue;
        int c = pc > 0 ? white : black;
        int t = pc > 0 ? pc : -pc;
        count[c][t]++;
        if (t == pawn) pawns_on_file[c][s % 8]++;
    }
    for (int t = pawn; t <= queen; t++) x[t - 1] = (float)(count[white][t] - count[black][t]);
    x[5] = (float)((count[white][bishop] >= 2) - (count[black][bishop] >= 2));
    x[6] = (mobility(p, white) - mobility(p, black)) / 10.0f;

    for (int s = 0; s < 64; s++) {
        int8_t pc = p.sq[s];
        int t = pc > 0 ? pc : -pc;
        if (!pc) continue;
        int c = pc > 0 ? white : black;
        float sgn = c == white ? 1.0f : -1.0f;
        int r = s / 8, fl = s % 8;
        if (t == pawn) {
            int adv = c == white ? r - 1 : 6 - r;
            x[7] += sgn * adv / 8.0f;
            // passed: no enemy pawn in front on this file or the next ones
            bool passed = true;
            for (int ff = std::max(0, fl - 1); ff <= std::min(7, fl + 1) && passed; ff++) {
                for (int rr = 0; rr < 8; rr++) {
                    int8_t q = p.sq[rr * 8 + ff];
                    if (q != (c == white ? -pawn : pawn)) continue;
                    if ((c == white && rr > r) || (c == black && rr < r)) passed = false;
                }
            }
            if (passed) x[8] += sgn;
            bool left = fl > 0 && pawns_on_file[c][fl - 1];
            bool right = fl < 7 && pawns_on_file[c][fl + 1];
            if (!left && !right) x[10] += sgn;
        }
        if ((s == 27 || s == 28 || s == 35 || s == 36) && t != king) x[11] += sgn;
        if (t == rook && pawns_on_file[c][fl] == 0) x[13] += sgn * (pawns_on_file[c ^ 1][fl] == 0 ? 1.0f : 0.5f);
    }
    for (int c = 0; c < 2; c++) {
        float sgn = c == white ? 1.0f : -1.0f;
        for (int fl = 0; fl < 8; fl++)
            if (pawns_on_file[c][fl] > 1) x[9] += sgn * (pawns_on_file[c][fl] - 1);
        // own pawns right in front of the king
        int k = p.king_sq[c], kr = k / 8, kf = k % 8, dir = c == white ? 1 : -1;
        for (int ff = std::max(0, kf - 1); ff <= std::min(7, kf + 1); ff++) {
            for (int step = 1; step <= 2; step++) {
                int rr = kr + dir * step;
                if (rr < 0 || rr > 7) continue;
                if (p.sq[rr * 8 + ff] == (c == white ? pawn : -pawn)) x[12] += sgn;
            }
        }
    }
    float stm = p.side == white ? 1.0f : -1.0f;
    for (int i = 0; i < 14; i++) f[i] = x[i] * stm;
    f[14] = p.in_check() ? -1.0f : 0.0f;  // already from the side to move
    f[15] = 1.0f;                         // just being the one to move
}

float logit(const weights& w, const position& p) {
    float f[n_feat];
    features(p, f);
    float s = 0;
    for (int i = 0; i < n_feat; i++) s += w.w[i] * f[i];
    return s;
}

static int piece_value(int8_t pc) {
    static const int v[7] = {0, 1, 3, 3, 5, 9, 100};
    return v[pc > 0 ? pc : -pc];
}

float quiet_logit(position& p, const weights& w, float alpha, float beta, int depth) {
    float stand = logit(w, p);
    if (depth <= 0) return stand;
    if (stand >= beta) return stand;
    alpha = std::max(alpha, stand);
    std::vector<move> caps;
    p.gen_pseudo(caps, true);
    // most valuable victim, least valuable attacker first
    std::sort(caps.begin(), caps.end(), [&](const move& a, const move& b) {
        int va = piece_value(p.sq[a.to]) * 10 - piece_value(p.sq[a.from]);
        int vb = piece_value(p.sq[b.to]) * 10 - piece_value(p.sq[b.from]);
        return va > vb;
    });
    float best = stand;
    for (const move& m : caps) {
        if (!p.make(m)) continue;
        float v = -quiet_logit(p, w, -beta, -alpha, depth - 1);
        p.unmake();
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) break;
    }
    return best;
}

#ifndef USE_CUDA
void eval_batch(const float* feats, int n, const weights& w, float* out) {
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < n_feat; j++) s += w.w[j] * feats[i * n_feat + j];
        out[i] = sigmoid(s);
    }
}
#endif

}  // namespace chess
