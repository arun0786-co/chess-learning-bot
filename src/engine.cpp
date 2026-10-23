// learnbot - a chess bot that starts knowing nothing and learns from playing itself
//
//   learnbot perft [depth]                 count moves (to check the rules)
//   learnbot train [--gens 20 --games 200]  self play + learn, saves models/
//   learnbot match a.txt b.txt [--games 40]
//   learnbot uci [--weights models/best.txt]
//   learnbot play [--weights models/best.txt]   you vs the bot in the terminal
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "board.h"
#include "eval.h"
#include "mcts.h"
#include "train.h"

using namespace chess;

static std::map<std::string, std::string> parse_args(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) == 0) {
            a[k.substr(2)] = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "1";
        } else {
            a["_" + std::to_string(a.size())] = k;
        }
    }
    return a;
}

static int num(const std::map<std::string, std::string>& a, const std::string& k, int def) {
    auto it = a.find(k);
    return it == a.end() ? def : std::atoi(it->second.c_str());
}

static std::string str(const std::map<std::string, std::string>& a, const std::string& k, const std::string& def) {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

static search_opts opts_from(const std::map<std::string, std::string>& a, int playouts) {
    search_opts o;
    o.playouts = num(a, "playouts", playouts);
    if (a.count("c")) o.c = (float)std::atof(a.at("c").c_str());
    o.quiet = num(a, "quiet", o.quiet);
    o.batch = num(a, "batch", o.batch);
    return o;
}

static weights load_or_zero(const std::string& path) {
    weights w;
    if (!path.empty() && !w.load(path)) std::fprintf(stderr, "could not read %s, using zero weights\n", path.c_str());
    return w;
}

static int cmd_perft(int depth) {
    position p;
    for (int d = 1; d <= depth; d++) {
        auto t0 = std::chrono::steady_clock::now();
        auto n = perft(p, d);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("depth %d: %llu  (%.2fs)\n", d, (unsigned long long)n, s);
    }
    return 0;
}

static int cmd_train(const std::map<std::string, std::string>& a) {
    int gens = num(a, "gens", 20), games = num(a, "games", 200), match_games = num(a, "match", 80);
    int seed = num(a, "seed", 1);
    std::string dir = str(a, "models", "models");
    std::string log_path = str(a, "log", "training/log.csv");
    search_opts o = opts_from(a, 100);
    std::mt19937 rng(seed);

    weights best;  // starts at zero: it knows nothing, not even that a queen is worth more than a pawn
    if (a.count("from")) best = load_or_zero(a.at("from"));
    best.save(dir + "/gen_00.txt");

    std::ofstream log(log_path);
    log << "gen,games,positions,white_wins,black_wins,draws,avg_plies,loss,score_vs_best,elo,kept";
    for (int i = 0; i < n_feat; i++) log << "," << feat_names[i];
    log << "\n";

    std::vector<std::vector<sample>> window;
    for (int g = 1; g <= gens; g++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<sample> data;
        game_stats st = self_play(best, games, o, 200, rng, data);
        window.push_back(data);
        if (window.size() > 3) window.erase(window.begin());  // last 3 generations of games
        std::vector<sample> all;
        for (auto& d : window) all.insert(all.end(), d.begin(), d.end());

        float loss = 0;
        weights cand = fit(all, best, 400, 1e-4f, &loss);
        float score = match(cand, best, match_games, o, 200, rng);
        bool kept = score >= 0.55f;
        if (kept) best = cand;

        char name[64];
        std::snprintf(name, sizeof(name), "/gen_%02d.txt", g);
        cand.save(dir + name);
        best.save(dir + "/best.txt");

        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("gen %d: %d games (%dW %dB %dD, %.0f plies avg), %zu positions, loss %.4f, new vs best %.0f%% (%+.0f elo) %s  [%.0fs]\n",
                    g, st.games, st.white_wins, st.black_wins, st.draws, (double)st.plies / st.games, all.size(), loss,
                    score * 100, elo_from_score(score), kept ? "KEPT" : "dropped", secs);
        std::printf("        piece weights now: P %.2f  N %.2f  B %.2f  R %.2f  Q %.2f\n", best.w[0], best.w[1], best.w[2],
                    best.w[3], best.w[4]);
        std::fflush(stdout);
        log << g << "," << st.games << "," << all.size() << "," << st.white_wins << "," << st.black_wins << "," << st.draws
            << "," << (double)st.plies / st.games << "," << loss << "," << score << "," << elo_from_score(score) << ","
            << kept;
        for (int i = 0; i < n_feat; i++) log << "," << cand.w[i];
        log << "\n";
        log.flush();
    }
    return 0;
}

static int cmd_match(const std::map<std::string, std::string>& a) {
    weights wa = load_or_zero(str(a, "_0", "")), wb = load_or_zero(str(a, "_1", ""));
    search_opts o = opts_from(a, 100);
    std::mt19937 rng(num(a, "seed", 7));
    int games = num(a, "games", 40);
    float s = match(wa, wb, games, o, 200, rng);
    std::printf("%s vs %s: %.1f / %d = %.0f%%  (%+.0f elo)\n", str(a, "_0", "zero").c_str(), str(a, "_1", "zero").c_str(),
                s * games, games, s * 100, elo_from_score(s));
    return 0;
}

static int think_time(const std::string& line, int side) {
    std::istringstream in(line);
    std::string t;
    int movetime = 0, wtime = -1, btime = -1, winc = 0, binc = 0, togo = 30;
    while (in >> t) {
        if (t == "movetime") in >> movetime;
        else if (t == "wtime") in >> wtime;
        else if (t == "btime") in >> btime;
        else if (t == "winc") in >> winc;
        else if (t == "binc") in >> binc;
        else if (t == "movestogo") in >> togo;
    }
    if (movetime > 0) return movetime;
    int left = side == white ? wtime : btime;
    int inc = side == white ? winc : binc;
    if (left < 0) return 1000;
    return std::max(10, left / std::max(togo, 10) + inc / 2 - 20);
}

static int cmd_uci(const std::map<std::string, std::string>& a) {
    weights w = load_or_zero(str(a, "weights", "models/best.txt"));
    position p;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci") {
            std::cout << "id name learnbot\nid author Arun\nuciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            p.set_fen(start_fen);
        } else if (cmd == "position") {
            std::string kind, tok;
            in >> kind;
            if (kind == "startpos") {
                p.set_fen(start_fen);
                in >> tok;  // "moves" or nothing
            } else if (kind == "fen") {
                std::string fen, part;
                while (in >> part && part != "moves") fen += (fen.empty() ? "" : " ") + part;
                p.set_fen(fen);
            }
            while (in >> tok) {
                move m;
                if (p.parse_move(tok, m)) p.make(m);
            }
        } else if (cmd == "go") {
            search_opts o = opts_from(a, 1 << 30);
            o.movetime_ms = think_time(line, p.side);
            if (line.find(" nodes ") != std::string::npos) {
                std::istringstream g(line.substr(line.find(" nodes ") + 7));
                g >> o.playouts;
                o.movetime_ms = 0;
            }
            search_result r = mcts(p, w, o);
            std::cout << "info nodes " << r.playouts << " score cp " << (int)(400 * std::log10(std::max(r.value, 0.001f) / std::max(1 - r.value, 0.001f)))
                      << std::endl;
            std::vector<move> legal;
            p.gen_legal(legal);
            std::cout << "bestmove " << (legal.empty() ? std::string("0000") : p.move_str(r.best)) << std::endl;
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}

static int cmd_play(const std::map<std::string, std::string>& a) {
    weights w = load_or_zero(str(a, "weights", "models/best.txt"));
    search_opts o = opts_from(a, 1 << 30);
    o.movetime_ms = num(a, "ms", 1500);
    position p;
    int you = str(a, "color", "white") == "black" ? black : white;
    std::printf("you play %s. type moves like e2e4, or quit\n", you == white ? "white" : "black");
    while (true) {
        bool over;
        float r = game_over(p, over);
        std::printf("\n%s\n", p.pretty().c_str());
        if (over) {
            if (r == 0.5f) std::printf("draw\n");
            else std::printf("%s wins\n", (p.side == white) == (r == 1.0f) ? "white" : "black");
            return 0;
        }
        if (p.side == you) {
            std::string s;
            std::printf("your move: ");
            if (!(std::cin >> s) || s == "quit") return 0;
            move m;
            if (!p.parse_move(s, m)) {
                std::printf("not a legal move\n");
                continue;
            }
            p.make(m);
        } else {
            search_result res = mcts(p, w, o);
            std::printf("bot plays %s  (thinks it wins %.0f%%)\n", p.move_str(res.best).c_str(), res.value * 100);
            p.make(res.best);
        }
    }
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "help";
    auto a = parse_args(argc, argv, 2);
    if (cmd == "perft") return cmd_perft(argc > 2 ? std::atoi(argv[2]) : 5);
    if (cmd == "train") return cmd_train(a);
    if (cmd == "match") return cmd_match(a);
    if (cmd == "uci") return cmd_uci(a);
    if (cmd == "play") return cmd_play(a);
    std::printf("usage: learnbot perft|train|match|uci|play  (see the readme)\n");
    return cmd == "help" ? 0 : 1;
}
