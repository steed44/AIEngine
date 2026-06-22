#include "patchcore/tiered_memory_bank.h"
#include "patchcore/memory_manager.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

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

    uint32_t magic = *reinterpret_cast<const uint32_t*>(mmapPtr_);
    if (magic != kMagic) {
        Clear();
        return Status{StatusCode::ErrorModelLoad, "invalid magic in memory bank: " + path};
    }

    num_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(
        static_cast<const char*>(mmapPtr_) + 4));
    dim_ = static_cast<int>(*reinterpret_cast<const uint32_t*>(
        static_cast<const char*>(mmapPtr_) + 8));

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

    float* hostBuf = new float[num_ * dim_];
    for (int i = 0; i < num_; i++) {
        const float* src = MmapFeatureAt(mmapPtr_, i, dim_);
        std::memcpy(hostBuf + i * dim_, src, dim_ * sizeof(float));
    }

    cudaError_t err = cudaMemcpy(gpu, hostBuf, bytes, cudaMemcpyHostToDevice);
    delete[] hostBuf;

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

    int maxRow = 0, maxCol = 0;
    for (auto& q : queries) {
        maxRow = std::max(maxRow, q.patchRow + 1);
        maxCol = std::max(maxCol, q.patchCol + 1);
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

std::vector<float> TieredMemoryBank::ComputeOnGPU(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {

    int M = static_cast<int>(queries.size());
    if (M == 0) return {};

    MemoryManager::GetInstance().Touch(gpuAllocId_);

    int maxRow = 0, maxCol = 0;
    for (auto& q : queries) {
        maxRow = std::max(maxRow, q.patchRow + 1);
        maxCol = std::max(maxCol, q.patchCol + 1);
    }
    if (maxRow == 0 || maxCol == 0) return {};

    extern "C" void BatchL2DistanceGPU(
        const float* queries, int M, int D,
        const float* bank, int N,
        float* outBestDists);

    // 构建稠密查询矩阵 M×D（主机端）
    float* hostQ = new float[M * dim_];
    for (int i = 0; i < M; i++) {
        std::memcpy(hostQ + i * dim_, queries[i].features.data(),
                    dim_ * sizeof(float));
    }

    float* devQ = nullptr;
    float* devDist = nullptr;
    size_t qBytes = static_cast<size_t>(M) * dim_ * sizeof(float);

    cudaMalloc(&devQ, qBytes);
    cudaMalloc(&devDist, M * sizeof(float));
    cudaMemcpy(devQ, hostQ, qBytes, cudaMemcpyHostToDevice);

    BatchL2DistanceGPU(devQ, M, dim_, gpuData_, num_, devDist);

    float* hostDist = new float[M];
    cudaMemcpy(hostDist, devDist, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devQ);
    cudaFree(devDist);
    delete[] hostQ;

    // 按空间位置填入得分图
    cv::Mat scoreMap(maxRow, maxCol, CV_32F);
    for (int i = 0; i < M; i++) {
        scoreMap.at<float>(queries[i].patchRow, queries[i].patchCol) = hostDist[i];
    }
    delete[] hostDist;

    cv::Mat upsampled;
    cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

    std::vector<float> result(imgW * imgH);
    std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
    return result;
}

} // namespace aicore
