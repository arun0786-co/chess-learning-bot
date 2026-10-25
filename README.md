# chess learning bot (learnbot)

a chess bot in c++ that starts knowing **nothing** - it doesnt even know a queen is worth more than a pawn - and learns only by playing against itself.

## how it learns

1. **play** - the bot plays 200 games against itself. it uses monte carlo tree search (MCTS, like alphazero) and judges positions with its current weights.
2. **learn** - every position from those games gets the final result (win / draw / loss). then a simple logistic regression finds weights so that `sigmoid(weights . features)` guesses the result.
3. **test** - the new weights play 80 games vs the current best. only if they score 55% or more they become the new best.
4. repeat.

the features are simple things like how many pawns / knights / rooks each side has, mobility, passed pawns, king shelter... 16 numbers. at the start all weights are 0.

## results

20 generations, about 5 minutes on my laptop (M4 Pro). full log in `training/log.csv`.

| gen | new vs best | kept? |
| --- | --- | --- |
| 1 | 76% (+197 elo) | yes |
| 3 | 60% (+70 elo) | yes |
| 4 | 58% (+53 elo) | yes |
| 10 | 57% (+48 elo) | yes |
| others | 39 - 54% | no |

**final bot vs the zero bot: 90% over 200 games (about +377 elo).**

what it learned about the pieces (weights in `models/best.txt`):

| queen | rook | bishop | knight | pawn |
| --- | --- | --- | --- | --- |
| 0.49 | 0.25 | 0.13 | 0.05 | 0.02 |

nobody told it that, it found the order queen > rook > minor pieces > pawn by itself. the knight is a bit low compared to what humans think (3 like a bishop).

after gen 10 it stops getting better. with only 16 simple features and short games there is not much more to learn. a real neural net would be the next step (see todo).

## the rules engine

`src/board.cpp` has all the rules: castling, en passant, promotion, check, 50 move rule, repetition. it is checked with **perft** (counting all positions some moves deep and comparing with known numbers). all of them match, including the tricky "kiwipete" position. about 26 million positions per second.

## build

```
mkdir build && cd build
cmake ..
make
ctest          # runs perft + bot tests
```

no cmake? this works too:

```
clang++ -std=c++17 -O2 -Isrc -o learnbot src/engine.cpp src/board.cpp src/eval.cpp src/mcts.cpp src/train.cpp
```

## use

```
./learnbot train                         # self play + learn, saves models/ and training/log.csv
./learnbot match models/best.txt models/gen_00.txt --games 200
./learnbot play                          # you vs the bot in the terminal
./learnbot uci --weights models/best.txt # for chess guis like cutechess or arena
./learnbot perft 5
```

it speaks UCI, so my other project **chess-auto-agent** can make it play against other bots.

## cuda

`cuda/mcts_kernel.cu` judges a batch of positions on an nvidia gpu (one thread per position). build with `cmake -DUSE_CUDA=ON ..` and run with `--quiet 0 --batch 64`.

honest note: i dont have an nvidia gpu (mac), so the cuda part is **not tested**. and with only 16 weights the gpu would be slower than the cpu anyway because of copying data. it makes sense once the eval is a bigger net.

## files

```
src/board.*      rules, move generation, perft
src/eval.*       features, weights, capture search at the leaves
src/mcts.*       the tree search
src/train.*      self play, learning, matches
src/engine.cpp   command line + uci
cuda/            gpu batch eval
models/          weights of every generation (gen_00 = all zero)
training/        log of the training run
tests/           perft + bot tests
```

## todo

- small neural net instead of 16 features (then the gpu is useful)
- policy part like alphazero (which moves to look at first)
- play on lichess as a bot account
