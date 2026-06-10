#pragma once
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace aicore {

class EnginePool {
public:
    explicit EnginePool(size_t maxEngines = 4);

    Status Acquire(const std::string& modelId, const ModelInfo& info,
                   std::shared_ptr<IModelBackend>& engine);
    Status Release(const std::string& modelId,
                   std::shared_ptr<IModelBackend> engine);
    void Clear();

private:
    struct PoolEntry {
        std::vector<std::shared_ptr<IModelBackend>> free;
        ModelInfo info;
    };

    size_t maxEngines_;
    std::unordered_map<std::string, PoolEntry> pools_;
    std::mutex mutex_;
};

} // namespace aicore
