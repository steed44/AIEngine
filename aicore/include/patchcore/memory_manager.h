#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace aicore {

class MemoryManager {
public:
    static MemoryManager& GetInstance();

    float* TryAlloc(size_t bytes, uint64_t& outAllocId);
    void Free(uint64_t allocId);
    void Touch(uint64_t allocId);

    void SetBudget(size_t maxBytes);
    size_t Budget() const { return budget_; }
    size_t Used() const { return used_; }
    size_t Available() const;

    static size_t QueryTotalVRAM();

private:
    MemoryManager();
    ~MemoryManager();
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    bool EvictLRU(size_t bytesNeeded);

    struct AllocRecord {
        uint64_t id;
        float* ptr;
        size_t bytes;
        uint64_t lastTouch;
    };

    std::mutex mutex_;
    size_t budget_;
    size_t used_ = 0;
    uint64_t clock_ = 0;
    uint64_t nextId_ = 1;
    std::map<uint64_t, AllocRecord> allocs_;
};

} // namespace aicore
