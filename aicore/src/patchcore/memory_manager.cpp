#include "patchcore/memory_manager.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace aicore {

static constexpr float kDefaultBudgetRatio = 0.7f;

MemoryManager& MemoryManager::GetInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() {
    size_t total = QueryTotalVRAM();
    budget_ = static_cast<size_t>(total * kDefaultBudgetRatio);
}

MemoryManager::~MemoryManager() {
    for (auto& [id, rec] : allocs_) {
        cudaFree(rec.ptr);
    }
    allocs_.clear();
}

size_t MemoryManager::QueryTotalVRAM() {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return 8ULL * 1024 * 1024 * 1024;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return 8ULL * 1024 * 1024 * 1024;

    return prop.totalGlobalMem;
}

float* MemoryManager::TryAlloc(size_t bytes, uint64_t& outAllocId) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (used_ + bytes > budget_) {
        if (!EvictLRU(bytes)) return nullptr;
    }

    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;

    uint64_t id = nextId_++;
    allocs_[id] = {id, ptr, bytes, clock_++};

    used_ += bytes;
    outAllocId = id;
    return ptr;
}

void MemoryManager::Free(uint64_t allocId) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocs_.find(allocId);
    if (it == allocs_.end()) return;

    cudaFree(it->second.ptr);
    used_ -= it->second.bytes;
    allocs_.erase(it);
}

void MemoryManager::Touch(uint64_t allocId) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocs_.find(allocId);
    if (it != allocs_.end()) {
        it->second.lastTouch = clock_++;
    }
}

void MemoryManager::SetBudget(size_t maxBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_ = maxBytes;
}

size_t MemoryManager::Available() const {
    if (used_ >= budget_) return 0;
    return budget_ - used_;
}

bool MemoryManager::EvictLRU(size_t bytesNeeded) {
    if (allocs_.empty()) return false;

    std::vector<AllocRecord*> sorted;
    sorted.reserve(allocs_.size());
    for (auto& [id, rec] : allocs_) {
        sorted.push_back(&rec);
    }

    std::sort(sorted.begin(), sorted.end(),
              [](const AllocRecord* a, const AllocRecord* b) {
                  return a->lastTouch < b->lastTouch;
              });

    size_t freed = 0;
    for (auto* rec : sorted) {
        if (freed >= bytesNeeded) break;
        cudaFree(rec->ptr);
        freed += rec->bytes;
        used_ -= rec->bytes;
        allocs_.erase(rec->id);
    }

    return freed >= bytesNeeded;
}

} // namespace aicore
