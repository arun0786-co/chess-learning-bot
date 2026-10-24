cmake_minimum_required(VERSION 3.18)
project(learnbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_CUDA "judge leaf positions on the gpu (needs an nvidia card + cuda)" OFF)

add_library(chesscore src/board.cpp src/eval.cpp src/mcts.cpp src/train.cpp)
target_include_directories(chesscore PUBLIC src)

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
  enable_language(CUDA)
  target_sources(chesscore PRIVATE cuda/mcts_kernel.cu)
  target_compile_definitions(chesscore PUBLIC USE_CUDA)
endif()

add_executable(learnbot src/engine.cpp)
target_link_libraries(learnbot chesscore)

enable_testing()
add_executable(perft_test tests/perft_test.cpp)
target_link_libraries(perft_test chesscore)
add_test(NAME perft COMMAND perft_test)

add_executable(bot_test tests/bot_test.cpp)
target_link_libraries(bot_test chesscore)
add_test(NAME bot COMMAND bot_test)
