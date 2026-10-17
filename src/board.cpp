#include "board.h"

#include <cctype>
#include <sstream>

namespace chess {

const char* start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// 10x12 mailbox, so moves that fall off the board land on -1
static const int mailbox[120] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7, -1,
    -1,  8,  9, 10, 11, 12, 13, 14, 15, -1,
    -1, 16, 17, 18, 19, 20, 21, 22, 23, -1,
    -1, 24, 25, 26, 27, 28, 29, 30, 31, -1,
    -1, 32, 33, 34, 35, 36, 37, 38, 39, -1,
    -1, 40, 41, 42, 43, 44, 45, 46, 47, -1,
    -1, 48, 49, 50, 51, 52, 53, 54, 55, -1,
    -1, 56, 57, 58, 59, 60, 61, 62, 63, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
static int mailbox64[64];

static const int knight_dirs[8] = {-21, -19, -12, -8, 8, 12, 19, 21};
static const int king_dirs[8] = {-11, -10, -9, -1, 1, 9, 10, 11};
static const int bishop_dirs[4] = {-11, -9, 9, 11};
static const int rook_dirs[4] = {-10, -1, 1, 10};

// castling rights that are lost when a piece moves from / to a square
static uint8_t castle_mask[64];

static uint64_t z_piece[13][64], z_side, z_castle[16], z_ep[8];

static int step(int s, int d) { return mailbox[mailbox64[s] + d]; }
static int idx(int8_t pc) { return pc + 6; }  // -6..6 -> 0..12

static bool tables_ready = false;

static void init_tables() {
    if (tables_ready) return;
    for (int i = 0; i < 120; i++)
        if (mailbox[i] >= 0) mailbox64[mailbox[i]] = i;
    for (int i = 0; i < 64; i++) castle_mask[i] = 15;
    castle_mask[0] = 15 & ~2;
    castle_mask[4] = 15 & ~3;
    castle_mask[7] = 15 & ~1;
    castle_mask[56] = 15 & ~8;
    castle_mask[60] = 15 & ~12;
    castle_mask[63] = 15 & ~4;
    uint64_t x = 0x9E3779B97F4A7C15ULL;  // splitmix64, fixed so hashes are the same every run
    auto next = [&]() {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (auto& row : z_piece)
        for (auto& v : row) v = next();
    z_side = next();
    for (auto& v : z_castle) v = next();
    for (auto& v : z_ep) v = next();
    tables_ready = true;
}

position::position() {
    init_tables();
    set_fen(start_fen);
}

uint64_t position::compute_hash() const {
    uint64_t h = 0;
    for (int s = 0; s < 64; s++)
        if (sq[s]) h ^= z_piece[idx(sq[s])][s];
    if (side == black) h ^= z_side;
    h ^= z_castle[castling];
    if (ep >= 0) h ^= z_ep[ep % 8];
    return h;
}

bool position::set_fen(const std::string& fen) {
    std::istringstream in(fen);
    std::string board, stm, cas, eps;
    in >> board >> stm >> cas >> eps >> halfmove >> fullmove;
    if (in.fail()) {
        halfmove = 0;
        fullmove = 1;
    }
    for (auto& v : sq) v = 0;
    int r = 7, f = 0;
    for (char c : board) {
        if (c == '/') {
            r--;
            f = 0;
        } else if (isdigit(c)) {
            f += c - '0';
        } else {
            const std::string names = "pnbrqk";
            auto p = names.find((char)tolower(c));
            if (p == std::string::npos || r < 0 || f > 7) return false;
            int8_t pc = (int8_t)(p + 1);
            sq[r * 8 + f] = isupper(c) ? pc : -pc;
            if (pc == king) king_sq[isupper(c) ? white : black] = r * 8 + f;
            f++;
        }
    }
    side = stm == "b" ? black : white;
    castling = 0;
    for (char c : cas) {
        if (c == 'K') castling |= 1;
        if (c == 'Q') castling |= 2;
        if (c == 'k') castling |= 4;
        if (c == 'q') castling |= 8;
    }
    ep = (eps.size() == 2) ? (int8_t)((eps[1] - '1') * 8 + (eps[0] - 'a')) : -1;
    history.clear();
    hash = compute_hash();
    return true;
}

std::string position::fen() const {
    std::string out;
    for (int r = 7; r >= 0; r--) {
        int gap = 0;
        for (int f = 0; f < 8; f++) {
            int8_t pc = sq[r * 8 + f];
            if (!pc) {
                gap++;
                continue;
            }
            if (gap) out += char('0' + gap), gap = 0;
            char c = " pnbrqk"[pc > 0 ? pc : -pc];
            out += pc > 0 ? (char)toupper(c) : c;
        }
        if (gap) out += char('0' + gap);
        if (r) out += '/';
    }
    out += side == white ? " w " : " b ";
    std::string cas;
    if (castling & 1) cas += 'K';
    if (castling & 2) cas += 'Q';
    if (castling & 4) cas += 'k';
    if (castling & 8) cas += 'q';
    out += cas.empty() ? "-" : cas;
    out += " ";
    if (ep >= 0) {
        out += char('a' + ep % 8);
        out += char('1' + ep / 8);
    } else {
        out += "-";
    }
    out += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
    return out;
}

bool position::attacked(int s, int by) const {
    int sign = by == white ? 1 : -1;
    // pawns: a white pawn attacks up, so look down from s
    int pd[2] = {by == white ? -9 : 9, by == white ? -11 : 11};
    for (int d : pd) {
        int t = step(s, d);
        if (t >= 0 && sq[t] == sign * pawn) return true;
    }
    for (int d : knight_dirs) {
        int t = step(s, d);
        if (t >= 0 && sq[t] == sign * knight) return true;
    }
    for (int d : king_dirs) {
        int t = step(s, d);
        if (t >= 0 && sq[t] == sign * king) return true;
    }
    for (int d : bishop_dirs) {
        for (int t = step(s, d); t >= 0; t = step(t, d)) {
            if (sq[t]) {
                if (sq[t] == sign * bishop || sq[t] == sign * queen) return true;
                break;
            }
        }
    }
    for (int d : rook_dirs) {
        for (int t = step(s, d); t >= 0; t = step(t, d)) {
            if (sq[t]) {
                if (sq[t] == sign * rook || sq[t] == sign * queen) return true;
                break;
            }
        }
    }
    return false;
}

static void add(std::vector<move>& out, int from, int to, int flags, int promo = 0) {
    move m;
    m.from = (uint8_t)from;
    m.to = (uint8_t)to;
    m.flags = (uint8_t)flags;
    m.promo = (uint8_t)promo;
    out.push_back(m);
}

static void add_pawn(std::vector<move>& out, int from, int to, int flags) {
    int r = to / 8;
    if (r == 0 || r == 7) {
        for (int p : {queen, rook, bishop, knight}) add(out, from, to, flags, p);
    } else {
        add(out, from, to, flags);
    }
}

void position::gen_pseudo(std::vector<move>& out, bool captures_only) const {
    out.clear();
    int sign = side == white ? 1 : -1;
    for (int s = 0; s < 64; s++) {
        int8_t pc = sq[s];
        if (!pc || (pc > 0) != (side == white)) continue;
        int type = pc * sign;
        if (type == pawn) {
            int fwd = side == white ? 8 : -8;
            int start_rank = side == white ? 1 : 6;
            int last_rank = side == white ? 6 : 1;
            int t = s + fwd;
            // pushes. promotions count as "noisy" too so quiet search sees them
            if (!sq[t] && (!captures_only || s / 8 == last_rank)) {
                add_pawn(out, s, t, 0);
                if (!captures_only && s / 8 == start_rank && !sq[t + fwd]) add(out, s, t + fwd, flag_double);
            }
            int cd[2] = {side == white ? 9 : -9, side == white ? 11 : -11};
            for (int d : cd) {
                int c = step(s, d);
                if (c < 0) continue;
                if (sq[c] && (sq[c] > 0) != (side == white)) add_pawn(out, s, c, flag_capture);
                else if (c == ep) add(out, s, c, flag_capture | flag_ep);
            }
            continue;
        }
        if (type == knight || type == king) {
            const int* dirs = type == knight ? knight_dirs : king_dirs;
            for (int i = 0; i < 8; i++) {
                int t = step(s, dirs[i]);
                if (t < 0) continue;
                if (!sq[t]) {
                    if (!captures_only) add(out, s, t, 0);
                } else if ((sq[t] > 0) != (side == white)) {
                    add(out, s, t, flag_capture);
                }
            }
        } else {
            std::vector<int> dirs;
            if (type == bishop || type == queen) dirs.insert(dirs.end(), bishop_dirs, bishop_dirs + 4);
            if (type == rook || type == queen) dirs.insert(dirs.end(), rook_dirs, rook_dirs + 4);
            for (int d : dirs) {
                for (int t = step(s, d); t >= 0; t = step(t, d)) {
                    if (!sq[t]) {
                        if (!captures_only) add(out, s, t, 0);
                        continue;
                    }
                    if ((sq[t] > 0) != (side == white)) add(out, s, t, flag_capture);
                    break;
                }
            }
        }
    }
    if (captures_only) return;
    // castling: squares between empty, king not in check and not passing through attack
    int them = side ^ 1;
    if (side == white) {
        if ((castling & 1) && !sq[5] && !sq[6] && sq[7] == rook && !attacked(4, them) && !attacked(5, them) && !attacked(6, them))
            add(out, 4, 6, flag_castle);
        if ((castling & 2) && !sq[1] && !sq[2] && !sq[3] && sq[0] == rook && !attacked(4, them) && !attacked(3, them) && !attacked(2, them))
            add(out, 4, 2, flag_castle);
    } else {
        if ((castling & 4) && !sq[61] && !sq[62] && sq[63] == -rook && !attacked(60, them) && !attacked(61, them) && !attacked(62, them))
            add(out, 60, 62, flag_castle);
        if ((castling & 8) && !sq[57] && !sq[58] && !sq[59] && sq[56] == -rook && !attacked(60, them) && !attacked(59, them) && !attacked(58, them))
            add(out, 60, 58, flag_castle);
    }
}

bool position::make(const move& m) {
    undo u;
    u.m = m;
    u.castling = castling;
    u.ep = ep;
    u.halfmove = halfmove;
    u.hash = hash;
    int8_t pc = sq[m.from];
    int sign = pc > 0 ? 1 : -1;
    int cap_sq = (m.flags & flag_ep) ? m.to - 8 * sign : m.to;
    u.captured = sq[cap_sq];
    history.push_back(u);

    if (u.captured) {
        hash ^= z_piece[idx(u.captured)][cap_sq];
        sq[cap_sq] = 0;
    }
    hash ^= z_piece[idx(pc)][m.from];
    sq[m.from] = 0;
    int8_t placed = m.promo ? (int8_t)(sign * m.promo) : pc;
    sq[m.to] = placed;
    hash ^= z_piece[idx(placed)][m.to];

    if (m.flags & flag_castle) {
        int rf, rt;
        if (m.to == 6) rf = 7, rt = 5;
        else if (m.to == 2) rf = 0, rt = 3;
        else if (m.to == 62) rf = 63, rt = 61;
        else rf = 56, rt = 59;
        int8_t r = sq[rf];
        sq[rf] = 0;
        sq[rt] = r;
        hash ^= z_piece[idx(r)][rf] ^ z_piece[idx(r)][rt];
    }
    if (pc * sign == king) king_sq[side] = m.to;

    hash ^= z_castle[castling];
    castling &= castle_mask[m.from] & castle_mask[m.to];
    hash ^= z_castle[castling];

    if (ep >= 0) hash ^= z_ep[ep % 8];
    ep = (m.flags & flag_double) ? (int8_t)(m.from + 8 * sign) : -1;
    if (ep >= 0) hash ^= z_ep[ep % 8];

    halfmove = (pc * sign == pawn || u.captured) ? 0 : halfmove + 1;
    if (side == black) fullmove++;
    side ^= 1;
    hash ^= z_side;

    if (attacked(king_sq[side ^ 1], side)) {
        unmake();
        return false;
    }
    return true;
}

void position::unmake() {
    undo u = history.back();
    history.pop_back();
    const move& m = u.m;
    side ^= 1;
    if (side == black) fullmove--;
    int8_t placed = sq[m.to];
    int sign = placed > 0 ? 1 : -1;
    int8_t pc = m.promo ? (int8_t)(sign * pawn) : placed;
    sq[m.from] = pc;
    sq[m.to] = 0;
    int cap_sq = (m.flags & flag_ep) ? m.to - 8 * sign : m.to;
    sq[cap_sq] = u.captured;
    if (m.flags & flag_castle) {
        int rf, rt;
        if (m.to == 6) rf = 7, rt = 5;
        else if (m.to == 2) rf = 0, rt = 3;
        else if (m.to == 62) rf = 63, rt = 61;
        else rf = 56, rt = 59;
        sq[rf] = sq[rt];
        sq[rt] = 0;
    }
    if (pc * sign == king) king_sq[side] = m.from;
    castling = u.castling;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
}

void position::gen_legal(std::vector<move>& out) {
    std::vector<move> all;
    gen_pseudo(all);
    out.clear();
    for (const move& m : all) {
        if (make(m)) {
            unmake();
            out.push_back(m);
        }
    }
}

int position::repetitions() const {
    int n = 0;
    // only positions since the last pawn move / capture can repeat
    int back = std::min<int>(halfmove, (int)history.size());
    for (int i = 2; i <= back; i += 2)
        if (history[history.size() - i].hash == hash) n++;
    return n;
}

bool position::insufficient() const {
    int minors = 0;
    for (int s = 0; s < 64; s++) {
        int t = sq[s] > 0 ? sq[s] : -sq[s];
        if (t == pawn || t == rook || t == queen) return false;
        if (t == knight || t == bishop) minors++;
    }
    return minors <= 1;
}

std::string position::move_str(const move& m) const {
    std::string s;
    s += char('a' + m.from % 8);
    s += char('1' + m.from / 8);
    s += char('a' + m.to % 8);
    s += char('1' + m.to / 8);
    if (m.promo) s += " pnbrqk"[m.promo];
    return s;
}

bool position::parse_move(const std::string& s, move& out) {
    std::vector<move> legal;
    gen_legal(legal);
    for (const move& m : legal) {
        if (move_str(m) == s) {
            out = m;
            return true;
        }
    }
    return false;
}

std::string position::pretty() const {
    std::string out;
    for (int r = 7; r >= 0; r--) {
        out += char('1' + r);
        out += "  ";
        for (int f = 0; f < 8; f++) {
            int8_t pc = sq[r * 8 + f];
            char c = pc ? " pnbrqk"[pc > 0 ? pc : -pc] : '.';
            out += pc > 0 ? (char)toupper(c) : c;
            out += ' ';
        }
        out += '\n';
    }
    out += "   a b c d e f g h\n";
    return out;
}

uint64_t perft(position& p, int depth) {
    if (depth == 0) return 1;
    std::vector<move> moves;
    p.gen_pseudo(moves);
    uint64_t n = 0;
    for (const move& m : moves) {
        if (!p.make(m)) continue;
        n += perft(p, depth - 1);
        p.unmake();
    }
    return n;
}

}  // namespace chess
