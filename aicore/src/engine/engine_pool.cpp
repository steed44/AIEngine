#include "engine/engine_pool.h"

namespace aicore {

EnginePool::EnginePool(size_t maxEngines)
    : maxEngines_(maxEngines) {}

Status EnginePool::Acquire(const std::string& modelId, const ModelInfo& info,
                           std::shared_ptr<IModelBackend>& engine) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = pools_[modelId];
    if (!entry.free.empty()) {
        engine = entry.free.back();
        entry.free.pop_back();
        return Status{};
    }
    auto newEngine = BackendFactory::Create(info.backend);
    if (!newEngine)
        return Status{StatusCode::ErrorInternal, "Failed to create backend"};
    auto s = newEngine->Load(info);
    if (!s) return s;
    engine = std::move(newEngine);
    return Status{};
}

Status EnginePool::Release(const std::string& modelId,
                           std::shared_ptr<IModelBackend> engine) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = pools_[modelId];
    if (entry.free.size() >= maxEngines_) {
        return Status{StatusCode::ErrorResourceExhaust, "Engine pool full"};
    }
    entry.free.push_back(std::move(engine));
    return Status{};
}

void EnginePool::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.clear();
}

} // namespace aicore
