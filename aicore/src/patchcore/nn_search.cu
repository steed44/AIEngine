// ============================================================
// nn_search.cu — CUDA 最近邻搜索内核
//
// 本文件实现了 BatchL2DistanceGPU CUDA kernel，用于 PatchCore
// 异常检测中的最近邻搜索。
//
// 算法原理：
//   对于每个 query feature（共 M 个），在 memory bank（共 N 个）
//   中找到 L2 距离最小的那个。朴素算法复杂度 O(M×N×D)，
//   其中 D 是特征维度。
//
// CUDA 优化策略：
//   1. 每个 block 处理一个 query，block 内线程并行搜索 bank
//   2. 使用 shared memory 做 block 内归约（reduce）找最小值
//   3. 使用 __restrict__ 指针告诉编译器数据不重叠，便于优化
// ============================================================

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

extern "C" {

/**
 * CUDA Kernel: 批量 L2 距离最近邻搜索
 *
 * 网格布局：gridDim.x = M（query 数量）
 * 线程布局：blockDim.x = kBlockSize（默认 256）
 *
 * 每个 block 处理一个 query：
 *   1. 线程并行遍历 bank 中的所有 feature
 *   2. 计算 L2 距离，维护局部最小值
 *   3. 使用 shared memory 做 block 内归约
 *   4. threadIdx=0 的线程写入结果
 *
 * @param queries     [M, D] 查询特征矩阵（设备内存）
 * @param bank        [N, D] 记忆库特征矩阵（设备内存）
 * @param outBestDists [M] 输出：每个 query 的最小距离
 * @param outBestIdxs  [M] 输出：每个 query 的最近邻索引
 * @param M           query 数量
 * @param N           bank 大小
 * @param D           特征维度
 */
__global__ void BatchL2Kernel(const float* __restrict__ queries,
                               const float* __restrict__ bank,
                               float* __restrict__ outBestDists,
                               int* __restrict__ outBestIdxs,
                               int M, int N, int D) {
    int qIdx = blockIdx.x;
    if (qIdx >= M) return;

    // 当前 query 的特征指针
    const float* q = queries + qIdx * D;
    float bestDist = FLT_MAX;
    int bestIdx = 0;

    // 线程并行遍历 bank：每个线程处理 bank 中的一个 feature
    // grid-stride loop 确保 N 不是 blockDim 的整数倍时也能正确处理
    for (int bIdx = threadIdx.x; bIdx < N; bIdx += blockDim.x) {
        const float* b = bank + bIdx * D;
        float dist = 0.0f;
        // 计算 L2 距离（平方和）
        for (int d = 0; d < D; d++) {
            float diff = q[d] - b[d];
            dist += diff * diff;
        }
        // 维护最小值
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = bIdx;
        }
    }

    // 使用 shared memory 做 block 内归约（reduce）找最小值
    extern __shared__ float shared[];
    int* sharedIdx = reinterpret_cast<int*>(shared + blockDim.x);

    shared[threadIdx.x] = bestDist;
    sharedIdx[threadIdx.x] = bestIdx;
    __syncthreads();

    // 二叉树归约：每轮将搜索范围减半
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

    // threadIdx=0 写入结果
    if (threadIdx.x == 0) {
        outBestDists[qIdx] = sqrtf(shared[0]);  // 开方得到真正的 L2 距离
        if (outBestIdxs) {
            outBestIdxs[qIdx] = sharedIdx[0];
        }
    }
}

/**
 * 调用 BatchL2Kernel CUDA kernel
 *
 * @param queries     [M, D] 查询特征（主机内存，会被拷贝到 GPU）
 * @param M           query 数量
 * @param D           特征维度
 * @param bank        [N, D] 记忆库特征（设备内存）
 * @param N           bank 大小
 * @param outBestDists [M] 输出：最小距离
 * @param outBestIdxs  [M] 输出：最近邻索引（可为 nullptr）
 * @param stream      CUDA stream，支持异步执行
 */
void BatchL2DistanceGPU(const float* queries, int M, int D,
                         const float* bank, int N,
                         float* outBestDists, int* outBestIdxs,
                         cudaStream_t stream) {
    constexpr int kBlockSize = 256;  // 每个 block 的线程数
    int sharedBytes = kBlockSize * (sizeof(float) + sizeof(int));  // shared memory 大小
    // 启动 kernel：M 个 block，每个 block kBlockSize 个线程
    BatchL2Kernel<<<M, kBlockSize, sharedBytes, stream>>>(
        queries, bank, outBestDists, outBestIdxs, M, N, D);
}

} // extern "C"
