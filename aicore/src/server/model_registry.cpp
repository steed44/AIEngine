// 模型注册表 — 多版本模型管理 + 引用计数 + LRU 淘汰
//
// 设计要点：
//   通过 shared_ptr + 原子引用计数（refCount）追踪活跃推理请求。
//   GetActive 递增 refCount，Release 递减。
//   Replace 原子交换 backend 指针，等待旧引用归零后释放。
//   EvictLRU 按 lastUsedTime 排序，淘汰最久未使用的模型，用于显存压力管理。
#include "server/model_registry.h"
#include <algorithm>
#include <sstream>

namespace aicore {

// 获取模型槽位（refCount +1），推理结束后必须 Release
// 返回 shared_ptr 持有 ModelSlot，即使被 Replace 也能安全使用旧 backend
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

// 释放模型槽位（refCount -1），与 GetActive 配对使用
// refCount 降为 0 后 slot 可以被 EvictLRU 淘汰或 Replace 交换
void ModelRegistry::Release(const std::shared_ptr<ModelSlot>& slot) {
    if (slot) {
        slot->refCount.fetch_sub(1);
    }
}

bool ModelRegistry::Contains(const std::string& name) const {
    std::shared_lock lock(rwLock_);
    return slots_.find(name) != slots_.end();
}

int ModelRegistry::GetVersion(const std::string& name) const {
    std::shared_lock lock(rwLock_);
    auto it = slots_.find(name);
    return it != slots_.end() ? it->second->version : 0;
}

// 热替换模型后端：等待旧推理完成后原子交换 backend 指针
// 原子安全：持有 swapMutex 时等待 refCount 归零，确保无活跃推理
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

// LRU 淘汰：释放最久未使用的模型以回收显存
// 仅淘汰 refCount == 0 的模型（无活跃推理时才能安全释放）
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

// 列出所有已加载模型，返回 JSON 数组格式
// 每个元素包含 name/version/refCount/vramMB 字段
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