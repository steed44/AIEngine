#include "server/model_registry.h"
#include <algorithm>
#include <sstream>

namespace aicore {

std::shared_ptr<ModelSlot> ModelRegistry::GetActive(const std::string& name) {
    std::shared_lock lock(rwLock_);
    auto it = slots_.find(name);
    if (it == slots_.end()) return nullptr;
    auto slot = it->second;
    slot->refCount.fetch_add(1);
    slot->lastUsedTime.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    return slot;
}

Status ModelRegistry::Replace(const std::string& name,
                               std::unique_ptr<IModelBackend> newBackend,
                               size_t vramMB, int newVersion) {
    std::unique_lock lock(rwLock_);
    std::shared_ptr<ModelSlot> oldSlot;

    auto it = slots_.find(name);
    if (it != slots_.end()) {
        oldSlot = it->second;
        std::unique_lock swapLock(oldSlot->swapMutex);
        // 等待旧后端 refCount 降为 0
        while (oldSlot->refCount.load() > 0) {
            swapLock.unlock();
            std::this_thread::yield();
            swapLock.lock();
        }
        // 原子交换
        oldSlot->backend = std::move(newBackend);
        oldSlot->version = newVersion;
        oldSlot->vramMB = vramMB;
        oldSlot->lastUsedTime.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    } else {
        auto newSlot = std::make_shared<ModelSlot>();
        newSlot->modelName = name;
        newSlot->version = newVersion;
        newSlot->backend = std::move(newBackend);
        newSlot->vramMB = vramMB;
        slots_[name] = newSlot;
    }
    return Status{};
}

Status ModelRegistry::EvictLRU(size_t neededMB) {
    std::unique_lock lock(rwLock_);
    std::vector<std::pair<uint64_t, std::string>> candidates;
    for (auto& [name, slot] : slots_) {
        if (slot->refCount.load() == 0) {
            candidates.emplace_back(slot->lastUsedTime.load(), name);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    size_t freed = 0;
    for (auto& [time, name] : candidates) {
        if (freed >= neededMB) break;
        auto it = slots_.find(name);
        if (it == slots_.end()) continue;
        freed += it->second->vramMB;
        slots_.erase(it);
    }
    if (freed < neededMB) {
        return Status{StatusCode::ErrorResourceExhaust,
            "EvictLRU: only freed " + std::to_string(freed) + "MB, need " + std::to_string(neededMB) + "MB"};
    }
    return Status{};
}

void ModelRegistry::Unload(const std::string& name) {
    std::unique_lock lock(rwLock_);
    slots_.erase(name);
}

std::string ModelRegistry::List() const {
    std::shared_lock lock(rwLock_);
    std::ostringstream json;
    json << "[";
    bool first = true;
    for (auto& [name, slot] : slots_) {
        if (!first) json << ",";
        first = false;
        json << "{\"name\":\"" << name << "\","
             << "\"version\":" << slot->version << ","
             << "\"refCount\":" << slot->refCount.load() << ","
             << "\"vramMB\":" << slot->vramMB
             << "}";
    }
    json << "]";
    return json.str();
}

} // namespace aicore