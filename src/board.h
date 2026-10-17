// the chess rules. 64 squares, a1 = 0, h8 = 63.
// pieces are signed: + white, - black. 1 pawn 2 knight 3 bishop 4 rook 5 queen 6 king
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

enum { white = 0, black = 1 };
enum { empty = 0, pawn = 1, knight = 2, bishop = 3, rook = 4, queen = 5, king = 6 };
enum { flag_capture = 1, flag_ep = 2, flag_castle = 4, flag_double = 8 };

struct move {
    uint8_t from = 0, to = 0, promo = 0, flags = 0;
    bool operator==(const move& o) const { return from == o.from && to == o.to && promo == o.promo; }
};

struct undo {
    move m;
    int8_t captured;
    uint8_t castling;
    int8_t ep;
    int halfmove;
    uint64_t hash;
};

struct position {
    int8_t sq[64] = {0};
    int side = white;
    uint8_t castling = 0;  // 1 = white short, 2 = white long, 4 = black short, 8 = black long
    int8_t ep = -1;        // square where an en passant capture can land, or -1
    int halfmove = 0;
    int fullmove = 1;
    uint64_t hash = 0;
    int king_sq[2] = {4, 60};
    std::vector<undo> history;

    position();
    bool set_fen(const std::string& fen);
    std::string fen() const;

    void gen_pseudo(std::vector<move>& out, bool captures_only = false) const;
    void gen_legal(std::vector<move>& out);
    bool make(const move& m);  // false = illegal (own king left in check), already undone
    void unmake();

    bool attacked(int s, int by) const;
    bool in_check() const { return attacked(king_sq[side], side ^ 1); }
    int repetitions() const;
    bool insufficient() const;

    std::string move_str(const move& m) const;
    bool parse_move(const std::string& s, move& out);
    uint64_t compute_hash() const;
    std::string pretty() const;
};

extern const char* start_fen;
uint64_t perft(position& p, int depth);

}  // namespace chess
