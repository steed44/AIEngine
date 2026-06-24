#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

extern "C" {

__global__ void BatchL2Kernel(const float* __restrict__ queries,
                               const float* __restrict__ bank,
                               float* __restrict__ outBestDists,
                               int* __restrict__ outBestIdxs,
                               int M, int N, int D) {
    int qIdx = blockIdx.x;
    if (qIdx >= M) return;

    const float* q = queries + qIdx * D;
    float bestDist = FLT_MAX;
    int bestIdx = 0;

    for (int bIdx = threadIdx.x; bIdx < N; bIdx += blockDim.x) {
        const float* b = bank + bIdx * D;
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = q[d] - b[d];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = bIdx;
        }
    }

    extern __shared__ float shared[];
    int* sharedIdx = reinterpret_cast<int*>(shared + blockDim.x);

    shared[threadIdx.x] = bestDist;
    sharedIdx[threadIdx.x] = bestIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float d1 = shared[threadIdx.x];
            float d2 = shared[threadIdx.x + s];
            if (d2 < d1) {
                shared[threadIdx.x] = d2;
                sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        outBestDists[qIdx] = sqrtf(shared[0]);
        if (outBestIdxs) {
            outBestIdxs[qIdx] = sharedIdx[0];
        }
    }
}

void BatchL2DistanceGPU(const float* queries, int M, int D,
                         const float* bank, int N,
                         float* outBestDists, int* outBestIdxs) {
    constexpr int kBlockSize = 256;
    int sharedBytes = kBlockSize * (sizeof(float) + sizeof(int));
    BatchL2Kernel<<<M, kBlockSize, sharedBytes>>>(queries, bank, outBestDists, outBestIdxs, M, N, D);
}

} // extern "C"
