// GPU 显存管理器实现 — 带 LRU 驱逐策略的内存池
// 管理 cudaMalloc/cudaFree，超出预算时自动回收最久未使用的分配
#include "patchcore/memory_manager.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace aicore {

// 默认 GPU 显存使用比例（总显存 70%，避免占满导致其他进程 OOM）
static constexpr float kDefaultBudgetRatio = 0.7f;

// 获取全局单例
MemoryManager& MemoryManager::GetInstance() {
    static MemoryManager instance;
    return instance;
}

// 构造函数：查询 GPU 总显存，设定预算为 70%
MemoryManager::MemoryManager() {
    size_t total = QueryTotalVRAM();
    budget_ = static_cast<size_t>(total * kDefaultBudgetRatio);
}

// 析构函数：释放所有未手动释放的显存
MemoryManager::~MemoryManager() {
    for (auto& [id, rec] : allocs_) {
        cudaFree(rec.ptr);
    }
    allocs_.clear();
}

// 查询当前 GPU 设备的总显存大小（单位：字节）
// 查询失败时回退到 8GB 默认值，保证程序可继续运行
size_t MemoryManager::QueryTotalVRAM() {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return 8ULL * 1024 * 1024 * 1024;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return 8ULL * 1024 * 1024 * 1024;

    return prop.totalGlobalMem;
}

// 尝试分配指定字节的 GPU 显存
// 如果超出预算，按 LRU 顺序驱逐最旧的分配直到满足需求
// 驱逐失败或 cudaMalloc 失败时返回 nullptr
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

// 释放指定 ID 的显存分配
void MemoryManager::Free(uint64_t allocId) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocs_.find(allocId);
    if (it == allocs_.end()) return;

    cudaFree(it->second.ptr);
    used_ -= it->second.bytes;
    allocs_.erase(it);
}

// 更新分配的最后访问时间（LRU 驱逐优先级降低）
void MemoryManager::Touch(uint64_t allocId) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocs_.find(allocId);
    if (it != allocs_.end()) {
        it->second.lastTouch = clock_++;
    }
}

// 设置显存预算上限
void MemoryManager::SetBudget(size_t maxBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_ = maxBytes;
}

// 查询剩余可用显存
size_t MemoryManager::Available() const {
    if (used_ >= budget_) return 0;
    return budget_ - used_;
}

// LRU 驱逐实现：
// 1. 将所有分配按 lastTouch 升序排序（最旧的在前）
// 2. 依次释放直到释放空间 >= bytesNeeded
// 3. 返回是否满足需求
bool MemoryManager::EvictLRU(size_t bytesNeeded) {
    if (allocs_.empty()) return false;

    // 收集所有分配记录指针
    std::vector<AllocRecord*> sorted;
    sorted.reserve(allocs_.size());
    for (auto& [id, rec] : allocs_) {
        sorted.push_back(&rec);
    }

    // 按最后访问时间升序排序（LRU：最近最少使用的排前面）
    std::sort(sorted.begin(), sorted.end(),
              [](const AllocRecord* a, const AllocRecord* b) {
                  return a->lastTouch < b->lastTouch;
              });

    // 从最旧开始释放，直到满足需求
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
