// 模型注册表 — 多版本模型管理和安全热替换
//
// ModelSlot：
//   每个已加载模型对应一个 ModelSlot，包含：
//   - backend：IModelBackend 实例（唯一所有权）
//   - refCount：原子引用计数，追踪正在进行的推理请求数
//   - lastUsedTime：LRU 淘汰依据
//   - swapMutex：热替换时保证排他
//
// 安全热替换：
//   Replace 创建新 backend → 等待旧 backend 的 refCount 归零
//   → 原子交换指针 → 旧 backend 自然销毁。
//   正在执行的推理请求不受影响（仍持有旧 slot shared_ptr）。
//
// LRU 淘汰：
//   EvictLRU 按 lastUsedTime 升序排列，优先淘汰最久未使用的模型。
//   仅淘汰 refCount == 0 的模型（无活跃推理请求）。
#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include "backend/backend_factory.h"

namespace aicore {

// 模型槽位 — 管理单个模型实例的全生命周期
// 原子引用计数确保推理未完成时不会释放 backend
struct ModelSlot {
    std::string modelName;                    // 模型唯一名称
    int version = 0;                          // 模型版本号（热替换时递增）
    std::unique_ptr<IModelBackend> backend;   // 后端推理引擎实例（唯一所有权）
    std::atomic<int> refCount{0};             // 活跃引用计数（GetActive +1，Release -1）
    std::atomic<uint64_t> lastUsedTime{0};    // 最近使用时间戳（LRU 淘汰依据）
    size_t vramMB = 0;                        // 模型预估显存占用（单位 MB）
    std::mutex swapMutex;                     // 热替换互斥锁（等待 refCount 归零后交换）
};

class ModelRegistry {
public:
    std::shared_ptr<ModelSlot> GetActive(const std::string& name);
    void Release(const std::shared_ptr<ModelSlot>& slot);
    bool Contains(const std::string& name) const;
    int GetVersion(const std::string& name) const;
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