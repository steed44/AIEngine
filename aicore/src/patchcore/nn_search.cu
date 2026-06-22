#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

extern "C" {

__global__ void BatchL2Kernel(const float* __restrict__ queries,
                               const float* __restrict__ bank,
                               float* __restrict__ bestDists,
                               int M, int N, int D) {
    int qIdx = blockIdx.x;
    if (qIdx >= M) return;

    const float* q = queries + qIdx * D;
    float best = FLT_MAX;

    for (int bIdx = threadIdx.x; bIdx < N; bIdx += blockDim.x) {
        const float* b = bank + bIdx * D;
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = q[d] - b[d];
            dist += diff * diff;
        }
        if (dist < best) best = dist;
    }

    extern __shared__ float shared[];
    shared[threadIdx.x] = best;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float other = shared[threadIdx.x + s];
            shared[threadIdx.x] = fminf(shared[threadIdx.x], other);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bestDists[qIdx] = sqrtf(shared[0]);
    }
}

void BatchL2DistanceGPU(const float* queries, int M, int D,
                         const float* bank, int N,
                         float* outBestDists) {
    constexpr int kBlockSize = 256;
    int sharedBytes = kBlockSize * sizeof(float);
    BatchL2Kernel<<<M, kBlockSize, sharedBytes>>>(queries, bank, outBestDists, M, N, D);
}

} // extern "C"
