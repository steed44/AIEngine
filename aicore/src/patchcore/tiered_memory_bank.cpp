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

    std::vector<float> hostBuf(num_ * dim_);
    for (int i = 0; i < num_; i++) {
        const float* src = MmapFeatureAt(mmapPtr_, i, dim_);
        std::memcpy(hostBuf.data() + i * dim_, src, dim_ * sizeof(float));
    }

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

void TieredMemoryBank::DemoteToCPU() {
    if (tier_ != BankTier::kGPU) return;

    MemoryManager::GetInstance().Free(gpuAllocId_);
    gpuData_ = nullptr;
    gpuAllocId_ = 0;
    tier_ = BankTier::kCPU;
}

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

std::vector<float> TieredMemoryBank::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    if (queries.empty() || !mmapPtr_) return {};

    if (tier_ == BankTier::kGPU && gpuData_) {
        return ComputeOnGPU(queries, imgH, imgW);
    }

    return ComputeOnCPU(queries, imgH, imgW);
}

std::vector<float> TieredMemoryBank::ComputeOnCPU(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    if (queries.empty() || !mmapPtr_) return {};

    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

    // 单层: 直接计算得分图后上采样
    if (numLayers <= 1) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) return {};

        cv::Mat scoreMap(maxRow, maxCol, CV_32F);
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

        for (int r = 0; r < imgH; r++) {
            float* fr = fused.ptr<float>(r);
            float* ur = upsampled.ptr<float>(r);
            for (int c = 0; c < imgW; c++) {
                if (ur[c] > fr[c]) fr[c] = ur[c];
            }
        }
    }

    std::vector<float> result(imgW * imgH);
    std::copy(fused.begin<float>(), fused.end<float>(), result.begin());
    return result;
}

std::vector<float> TieredMemoryBank::ComputeOnGPU(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    int M = static_cast<int>(queries.size());
    if (M == 0) return {};

    MemoryManager::GetInstance().Touch(gpuAllocId_);

    // 一次 GPU 推理求所有查询距离 (batch 模式下效率最高)
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

    std::vector<float> hostDist(M);
    cudaMemcpy(hostDist.data(), devDist.Get(), M * sizeof(float), cudaMemcpyDeviceToHost);

    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

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

        for (int r = 0; r < imgH; r++) {
            float* fr = fused.ptr<float>(r);
            float* ur = upsampled.ptr<float>(r);
            for (int c = 0; c < imgW; c++) {
                if (ur[c] > fr[c]) fr[c] = ur[c];
            }
        }
    }

    std::vector<float> result(imgW * imgH);
    std::copy(fused.begin<float>(), fused.end<float>(), result.begin());
    return result;
}

} // namespace aicore
