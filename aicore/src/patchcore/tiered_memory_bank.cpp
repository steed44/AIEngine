#include "patchcore/tiered_memory_bank.h"
#include "patchcore/memory_manager.h"
#include "core/cuda_mem.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// 在 namespace 外部声明 CUDA kernel，满足 MSVC 链接规范要求
extern "C" void BatchL2DistanceGPU(
    const float* queries, int M, int D,
    const float* bank, int N,
    float* outBestDists, int* outBestIdxs = nullptr,
    cudaStream_t stream = nullptr);

// ============================================================
// tiered_memory_bank.cpp — 分级 MemoryBank 实现
// 功能：提供 GPU/CPU/磁盘三级特征存储，支持特征升级/降级迁移，
//       以及基于最近邻搜索的异常得分图计算。
//
// 三级存储架构：
//   kGPU (Hot) — 数据在显存中，GPU 并行计算距离，最快
//   kCPU (Warm) — 数据在系统内存，CPU 暴力搜索，中等速度
//   kDisk (Cold) — 数据仅 mmap 在磁盘，使用时按需加载
//
// 升级/降级策略：
//   PromoteToGPU: 从磁盘读到显存，用于高吞吐推理
//   DemoteToCPU:  释放显存，保留在 mmap（下次读可能触发 page fault）
//   DemoteToDisk: 释放显存 + 解锁 mmap 页，完全回到磁盘
//
// 异常得分图计算（NN Search + Score Map）：
//   1. 对每个 patch query，在 memory bank 中找最近邻（L2 距离）
//   2. 最近邻距离的平方根作为该 patch 的异常得分
//   3. 所有 patch 的得分组成特征网格得分图
//   4. 双线性插值上采样到原图尺寸
//   5. 多层特征：每层独立算得分图→上采样→逐像素取最大值融合
// ============================================================

namespace aicore {

namespace {

const float* MmapFeatureData(void* mmapPtr) {
    return reinterpret_cast<const float*>(
        static_cast<const char*>(mmapPtr) + 12);
}

int MmapFeatureStride(int dim) {
    return dim + 3;
}

const float* MmapFeatureAt(void* mmapPtr, int idx, int dim) {
    const float* data = MmapFeatureData(mmapPtr);
    return data + idx * MmapFeatureStride(dim);
}

} // anonymous namespace

TieredMemoryBank::~TieredMemoryBank() {
    Clear();
}

TieredMemoryBank::TieredMemoryBank(TieredMemoryBank&& other) noexcept
    : tier_(other.tier_)
    , path_(std::move(other.path_))
    , num_(other.num_)
    , dim_(other.dim_)
    , mmapPtr_(other.mmapPtr_)
    , mmapSize_(other.mmapSize_)
    , gpuData_(other.gpuData_)
    , gpuAllocId_(other.gpuAllocId_)
{
    other.tier_ = BankTier::kDisk;
    other.num_ = 0;
    other.dim_ = 0;
    other.mmapPtr_ = nullptr;
    other.mmapSize_ = 0;
    other.gpuData_ = nullptr;
    other.gpuAllocId_ = 0;
}

TieredMemoryBank& TieredMemoryBank::operator=(TieredMemoryBank&& other) noexcept {
    if (this != &other) {
        Clear();
        tier_ = other.tier_;
        path_ = std::move(other.path_);
        num_ = other.num_;
        dim_ = other.dim_;
        mmapPtr_ = other.mmapPtr_;
        mmapSize_ = other.mmapSize_;
        gpuData_ = other.gpuData_;
        gpuAllocId_ = other.gpuAllocId_;

        other.tier_ = BankTier::kDisk;
        other.num_ = 0;
        other.dim_ = 0;
        other.mmapPtr_ = nullptr;
        other.mmapSize_ = 0;
        other.gpuData_ = nullptr;
        other.gpuAllocId_ = 0;
    }
    return *this;
}

// 从 .bin 文件加载特征库（mmap 到磁盘层级）
// 文件格式（新格式）：
//   magic(4B) + version(4B) + num(4B) + dim(4B) + [hasNorm(1B)+mean/std] + features
// 文件格式（旧格式）：
//   magic(4B) + num(4B) + dim(4B) + features
//
// mmap 优势：不用把整个文件读进内存，OS 按需加载页面，
// 对于大型 memory bank（数 GB）特别重要。
Status TieredMemoryBank::Load(const std::string& path) {
    Clear();
    path_ = path;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        return Status{StatusCode::ErrorModelLoad, "cannot open file: " + path};
    }

    HANDLE hMapping = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
        CloseHandle(hFile);
        return Status{StatusCode::ErrorModelLoad, "cannot create file mapping: " + path};
    }

    void* addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return Status{StatusCode::ErrorModelLoad, "cannot map view of file: " + path};
    }

    mmapPtr_ = addr;
    mmapSize_ = GetFileSize(hFile, nullptr);
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status{StatusCode::ErrorModelLoad, "cannot open file: " + path};
    }

    struct stat st;
    fstat(fd, &st);
    mmapSize_ = st.st_size;

    void* addr = mmap(nullptr, mmapSize_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        return Status{StatusCode::ErrorModelLoad, "mmap failed: " + path};
    }
    mmapPtr_ = addr;
#endif

    const char* base = static_cast<const char*>(mmapPtr_);
    uint32_t magic = *reinterpret_cast<const uint32_t*>(base);
    if (magic == MemoryBank::kMagicNew) {
        // 新格式: magic(4) + version(4) + num(4) + dim(4) + hasNorm(1) + [mean/std] + features
        uint32_t version = *reinterpret_cast<const uint32_t*>(base + 4);
        if (version != MemoryBank::kCurrentVersion) {
            Clear();
            return Status{StatusCode::ErrorModelLoad,
                "unsupported memory bank version: " + std::to_string(version)};
        }
        num_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(base + 8));
        dim_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(base + 12));
    } else if (magic == kMagic) {
        // 旧格式: magic(4) + num(4) + dim(4) + features
        num_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(base + 4));
        dim_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(base + 8));
    } else {
        Clear();
        return Status{StatusCode::ErrorModelLoad, "invalid magic in memory bank: " + path};
    }

    tier_ = BankTier::kDisk;
    return Status{};
}

// 清理所有资源：释放 GPU 显存、取消 mmap 映射、重置状态
void TieredMemoryBank::Clear() {
    if (gpuData_) {
        MemoryManager::GetInstance().Free(gpuAllocId_);
        gpuData_ = nullptr;
        gpuAllocId_ = 0;
    }

    if (mmapPtr_) {
#ifdef _WIN32
        UnmapViewOfFile(mmapPtr_);
#else
        munmap(mmapPtr_, mmapSize_);
#endif
        mmapPtr_ = nullptr;
        mmapSize_ = 0;
    }

    tier_ = BankTier::kDisk;
    num_ = 0;
    dim_ = 0;
    path_.clear();
}

// 将特征库从磁盘升级到 GPU 显存
// 步骤：
//   1. 从 MemoryManager 申请显存（可能触发 LRU 驱逐）
//   2. 通过 mmap 指针逐条读取特征数据到 host buffer
//   3. cudaMemcpy H2D 将数据传输到显存
//   4. 设置 tier_ = kGPU
//
// 注意：mmap 文件 -> host buffer -> 显存 这条路不是零拷贝，
// 但对于推理场景（每个 feature 只读一次），性能足够。
// 未来可优化为 GPU Direct RDMA 或 cuFile 实现 GPU 直接读盘。
Status TieredMemoryBank::PromoteToGPU() {
    if (tier_ == BankTier::kGPU) return Status{};
    if (!mmapPtr_ || num_ <= 0 || dim_ <= 0) {
        return Status{StatusCode::ErrorInternal, "bank not loaded"};
    }

    size_t bytes = static_cast<size_t>(num_) * dim_ * sizeof(float);

    uint64_t allocId = 0;
    float* gpu = MemoryManager::GetInstance().TryAlloc(bytes, allocId);
    if (!gpu) {
        return Status{StatusCode::ErrorResourceExhaust,
                      "GPU OOM: cannot allocate bank (" + std::to_string(bytes / 1048576) + " MB)"};
    }

    // 单次 bulk memcpy：从 mmap 跳过 header 直接拷贝 N×D 个 float
    // 之前逐行 memcpy（O(N) 次调用）改为单次大块拷贝，性能提升显著
    std::vector<float> hostBuf(num_ * dim_);
    const float* src = MmapFeatureData(mmapPtr_);
    std::memcpy(hostBuf.data(), src, bytes);

    cudaError_t err = cudaMemcpy(gpu, hostBuf.data(), bytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        MemoryManager::GetInstance().Free(allocId);
        return Status{StatusCode::ErrorInternal, "cudaMemcpy H2D failed"};
    }

    gpuData_ = gpu;
    gpuAllocId_ = allocId;
    tier_ = BankTier::kGPU;
    return Status{};
}

// 降级到 CPU 内存：释放 GPU 显存，数据仍在 mmap 中
// 当其他任务需要显存时调用。数据保留在 mmap 映射区域，
// 下次访问时 OS 会通过缺页中断自动加载。
void TieredMemoryBank::DemoteToCPU() {
    if (tier_ != BankTier::kGPU) return;

    MemoryManager::GetInstance().Free(gpuAllocId_);
    gpuData_ = nullptr;
    gpuAllocId_ = 0;
    tier_ = BankTier::kCPU;
}

// 降级到磁盘：释放显存 + 通知 OS 可回收 mmap 页面
// Windows: VirtualUnlock 允许系统将 mmap 页换出到磁盘
// Linux:   madvise MADV_DONTNEED 通知内核可以优先回收这些页
// 注意：mmap 映射本身不解除，指针仍有效，只是页面不再驻留 RAM。
void TieredMemoryBank::DemoteToDisk() {
    if (tier_ == BankTier::kGPU) {
        MemoryManager::GetInstance().Free(gpuAllocId_);
        gpuData_ = nullptr;
        gpuAllocId_ = 0;
    }

#ifdef _WIN32
    if (mmapPtr_) {
        VirtualUnlock(mmapPtr_, mmapSize_);
    }
#else
    if (mmapPtr_) {
        madvise(mmapPtr_, mmapSize_, MADV_DONTNEED);
    }
#endif

    tier_ = BankTier::kDisk;
}

// 计算异常热力图入口
// 根据当前存储层级自动选择 CPU 或 GPU 计算路径：
//   kGPU → ComputeOnGPU（CUDA 批量距离计算）
//   kCPU/kDisk → ComputeOnCPU（暴力 NN 搜索，逐 patch 遍历 memory bank）
//
// @param queries  从 backbone 提取的 patch 特征列表
// @param imgH     原始图像高度（用于上采样到原图尺寸）
// @param imgW     原始图像宽度
// @return 异常热力图（平坦数组，大小 imgH × imgW，值越大越异常）
std::vector<float> TieredMemoryBank::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    if (queries.empty() || !mmapPtr_) return {};

    if (tier_ == BankTier::kGPU && gpuData_) {
        return ComputeOnGPU(queries, imgH, imgW);
    }

    return ComputeOnCPU(queries, imgH, imgW);
}

// CPU 路径：暴力最近邻搜索（Brute-Force NN）
//
// 算法说明：
//   对于每个 patch query q ∈ queries：
//     在 memory bank 的 N 个特征中找最近邻：
//       d(q) = min_i ||q - bank_i||^2          (L2 距离平方)
//     score(q) = sqrt(d(q))                    (转为欧氏距离)
//
//   得分图上采样：
//     feature grid (Hf × Wf) → bilinear → image size (H × W)
//     Hf = 特征图高度（如 28），Wf = 特征图宽度（如 28）
//
//   多层融合策略：
//     每层 backbone 独立计算得分图 → 上采样到原图尺寸
//     → 逐像素取最大值作为最终得分
//     （max-fusion：保留跨层最强的异常响应）
//
// 复杂度：O(M × N × D)，其中 M=patch 数，N=banksize，D=特征维度
std::vector<float> TieredMemoryBank::ComputeOnCPU(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    if (queries.empty() || !mmapPtr_) return {};

    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

    // 单层情况：直接在特征网格上计算得分图，然后上采样到原图尺寸
    // 单层路径更简单，避免了逐像素融合的开销。
    if (numLayers <= 1) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) return {};

        cv::Mat scoreMap(maxRow, maxCol, CV_32F);
        // OpenMP 并行：每个 query 的 NN 搜索独立，可并行归约
        // #pragma omp parallel for schedule(dynamic, 16)
        //   - 动态分块 16：每个线程处理 16 个 queries，负载均衡
        //   - scoreMap 写入不同行/列，无数据竞争
        // NOTE: 当前注释掉，编译时需确认 OpenMP 可用后再启用
        for (auto& q : queries) {
            float bestDist = std::numeric_limits<float>::max();
            for (int i = 0; i < num_; i++) {
                const float* bankFeat = MmapFeatureAt(mmapPtr_, i, dim_);
                float d = 0;
                for (int j = 0; j < dim_; j++) {
                    float diff = q.features[j] - bankFeat[j];
                    d += diff * diff;
                }
                if (d < bestDist) bestDist = d;
            }
            scoreMap.at<float>(q.patchRow, q.patchCol) = std::sqrt(bestDist);
        }
        cv::Mat upsampled;
        cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        std::vector<float> result(imgW * imgH);
        std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
        return result;
    }

    // 多层融合: 每层独立计算得分图 → 上采样 → 逐像素取最大值
    cv::Mat fused(imgH, imgW, CV_32F, cv::Scalar(0));

    for (int li = 0; li < numLayers; li++) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) continue;

        cv::Mat layerMap(maxRow, maxCol, CV_32F);
        // OpenMP 并行：每层内 query 间 NN 搜索独立
        // #pragma omp parallel for schedule(dynamic, 16)
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            float bestDist = std::numeric_limits<float>::max();
            for (int i = 0; i < num_; i++) {
                const float* bankFeat = MmapFeatureAt(mmapPtr_, i, dim_);
                float d = 0;
                for (int j = 0; j < dim_; j++) {
                    float diff = q.features[j] - bankFeat[j];
                    d += diff * diff;
                }
                if (d < bestDist) bestDist = d;
            }
            layerMap.at<float>(q.patchRow, q.patchCol) = std::sqrt(bestDist);
        }

        cv::Mat upsampled;
        cv::resize(layerMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        // CPU 路径：逐像素 max-fusion，用 cv::max 替代标量循环（SIMD 加速）
        cv::max(fused, upsampled, fused);
    }

    std::vector<float> result(imgW * imgH);
    std::copy(fused.begin<float>(), fused.end<float>(), result.begin());
    return result;
}

// GPU 路径：CUDA 批量最近邻搜索
//
// 与 CPU 路径的区别：
//   1. 所有 query 特征打包成一个 batch，一次性传到 GPU
//   2. 调用 BatchL2DistanceGPU CUDA kernel 并行计算所有距离
//   3. kernel 输出每个 query 到 bank 的最小距离（已取平方根？）
//   4. 结果拷回 CPU 组装得分图
//
// 优势：GPU 并行计算 O(M × N × D) 距离矩阵，
//       对大规模 memory bank（N > 10 万）性能优势显著。
//
// 注意：这里用 Touch() 更新 LRU 时间戳，防止在计算过程中
//       被 MemoryManager 的 LRU 驱逐策略误回收。
std::vector<float> TieredMemoryBank::ComputeOnGPU(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    int M = static_cast<int>(queries.size());
    if (M == 0) return {};

    // Touch 更新 LRU 时间戳，防止被 MemoryManager 驱逐
    MemoryManager::GetInstance().Touch(gpuAllocId_);

    // 将所有 query 特征打包为连续内存块，一次 H2D 传输
    // 批量传输比逐条传输效率高得多（减少 PCIe 传输次数）
    std::vector<float> hostQ(M * dim_);
    for (int i = 0; i < M; i++) {
        std::memcpy(hostQ.data() + i * dim_, queries[i].features.data(),
                    dim_ * sizeof(float));
    }

    size_t qBytes = static_cast<size_t>(M) * dim_ * sizeof(float);
    CudaMem devQ, devDist;
    if (devQ.Alloc(qBytes) != cudaSuccess ||
        devDist.Alloc(M * sizeof(float)) != cudaSuccess) {
        return {};
    }
    cudaMemcpy(devQ.Get(), hostQ.data(), qBytes, cudaMemcpyHostToDevice);
    BatchL2DistanceGPU(devQ.Get(), M, dim_, gpuData_, num_, devDist.Get());

    // GPU 算完距离后一次性 D2D 拷回（再次减少 PCIe 传输次数）
    std::vector<float> hostDist(M);
    cudaMemcpy(hostDist.data(), devDist.Get(), M * sizeof(float), cudaMemcpyDeviceToHost);

    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

    // 单层：直接把 GPU 距离结果填入得分图后上采样
    if (numLayers <= 1) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) return {};

        cv::Mat scoreMap(maxRow, maxCol, CV_32F);
        for (int i = 0; i < M; i++) {
            scoreMap.at<float>(queries[i].patchRow, queries[i].patchCol) = hostDist[i];
        }
        cv::Mat upsampled;
        cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        std::vector<float> result(imgW * imgH);
        std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
        return result;
    }

    // 多层融合: GPU 距离结果按 layerIdx 分组构建得分图
    cv::Mat fused(imgH, imgW, CV_32F, cv::Scalar(0));

    for (int li = 0; li < numLayers; li++) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) continue;

        cv::Mat layerMap(maxRow, maxCol, CV_32F);
        for (int i = 0; i < M; i++) {
            if (queries[i].layerIdx != li) continue;
            layerMap.at<float>(queries[i].patchRow, queries[i].patchCol) = hostDist[i];
        }

        cv::Mat upsampled;
        cv::resize(layerMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        // GPU 路径：逐像素 max-fusion，用 cv::max 替代标量循环（SIMD 加速）
        cv::max(fused, upsampled, fused);
    }

    std::vector<float> result(imgW * imgH);
    std::copy(fused.begin<float>(), fused.end<float>(), result.begin());
    return result;
}

} // namespace aicore
