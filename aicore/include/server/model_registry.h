#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include "backend/backend_factory.h"

namespace aicore {

struct ModelSlot {
    std::string modelName;
    int version = 0;
    std::unique_ptr<IModelBackend> backend;
    std::atomic<int> refCount{0};
    std::atomic<uint64_t> lastUsedTime{0};
    size_t vramMB = 0;
    std::mutex swapMutex;
};

class ModelRegistry {
public:
    std::shared_ptr<ModelSlot> GetActive(const std::string& name);
    Status Replace(const std::string& name, std::unique_ptr<IModelBackend> newBackend,
                   size_t vramMB, int newVersion);
    Status EvictLRU(size_t neededMB);
    void Unload(const std::string& name);
    std::string List() const;

private:
    mutable std::shared_mutex rwLock_;
    std::unordered_map<std::string, std::shared_ptr<ModelSlot>> slots_;
};

} // namespace aicore