// gpu version of eval_batch: judges many leaf positions of the search at once.
// one thread per position: dot product of features and weights, then sigmoid.
//
// only built with  cmake -DUSE_CUDA=ON  (needs an nvidia card + cuda toolkit).
// note: i have no nvidia gpu on my mac, so this part is NOT tested yet.
// with only 16 weights the gpu is slower than the cpu because of copying, it starts
// to make sense when the eval becomes a real neural net (see todo in the readme).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../src/eval.h"

namespace chess {

__global__ void eval_kernel(const float* feats, int n, const float* w, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int j = 0; j < n_feat; j++) s += w[j] * feats[i * n_feat + j];
    out[i] = 1.0f / (1.0f + expf(-s));
}

static float* d_feats = nullptr;
static float* d_w = nullptr;
static float* d_out = nullptr;
static int room = 0;  // how many positions the gpu buffers can hold

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cuda error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void eval_batch(const float* feats, int n, const weights& w, float* out) {
    if (n <= 0) return;
    if (!d_w) check(cudaMalloc(&d_w, n_feat * sizeof(float)), "malloc w");
    if (n > room) {
        cudaFree(d_feats);
        cudaFree(d_out);
        room = n * 2;
        check(cudaMalloc(&d_feats, (size_t)room * n_feat * sizeof(float)), "malloc feats");
        check(cudaMalloc(&d_out, (size_t)room * sizeof(float)), "malloc out");
    }
    // weights change during training, so send them every time (only 16 floats)
    check(cudaMemcpy(d_w, w.w, n_feat * sizeof(float), cudaMemcpyHostToDevice), "copy w");
    check(cudaMemcpy(d_feats, feats, (size_t)n * n_feat * sizeof(float), cudaMemcpyHostToDevice), "copy feats");
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    eval_kernel<<<blocks, threads>>>(d_feats, n, d_w, d_out);
    check(cudaGetLastError(), "kernel");
    check(cudaMemcpy(out, d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
}

}  // namespace chess
