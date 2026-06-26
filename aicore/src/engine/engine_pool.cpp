// ============================================================
// engine_pool.cpp — 后端引擎对象池
// 按 modelId 分组缓存 IModelBackend 实例，避免重复创建和加载模型。
// 池满时释放多余引擎，支持跨管线复用相同模型的后端资源。
// ============================================================
//
// ---- 引擎池设计 ----
// 类似数据库连接池模式：预创建多个推理引擎实例，
// 按需分配（Acquire）和归还（Release）。
//
// 为什么需要引擎池而非每次创建新引擎：
//   1. 模型加载开销大（TensorRT 引擎构建可能耗时数秒）
//   2. GPU 显存分配开销大
//   3. 避免反复创建/销毁 cuDNN/cuBLAS handle
//
// 当前实现是简单的池，适合单模型多路并发的场景。
// 后续改进方向：支持多模型热替换、动态扩容。

#include "engine/engine_pool.h"

namespace aicore {

/**
 * 创建引擎池
 * @param maxEngines 每个模型最多缓存的引擎实例数
 */
EnginePool::EnginePool(size_t maxEngines)
    : maxEngines_(maxEngines) {}

/**
 * 从池中获取一个引擎实例
 * 优先返回空闲列表中的缓存引擎；若无缓存则创建新实例并加载模型
 * @param modelId 模型唯一标识
 * @param info    模型加载信息（路径、后端类型、设备等）
 * @param engine  输出参数，获取到的后端引擎共享指针
 * @return Status 成功返回空 Status，创建/加载失败返回错误
 */
Status EnginePool::Acquire(const std::string& modelId, const ModelInfo& info,
                           std::shared_ptr<IModelBackend>& engine) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = pools_[modelId];
    // 优先复用空闲引擎（从池尾取，避免 vector 移位成本）
    if (!entry.free.empty()) {
        engine = entry.free.back();
        entry.free.pop_back();
        return Status{};
    }
    // 无空闲引擎时创建新的（首次加载或池中实例全部被占用）
    auto newEngine = BackendFactory::Create(info.backend);
    if (!newEngine)
        return Status{StatusCode::ErrorInternal, "Failed to create backend"};
    auto s = newEngine->Load(info);
    if (!s) return s;
    engine = std::move(newEngine);
    return Status{};
}

/**
 * 将引擎实例归还到池中
 * 归还后引擎重新变为空闲状态，可被其他 Acquire 调用复用
 * 若该模型的缓存已达上限则丢弃（不报错，仅返回资源耗尽状态）
 * @param modelId 模型唯一标识
 * @param engine  待归还的引擎共享指针
 * @return Status 成功或资源耗尽（池满）均返回，失败返回错误
 */
Status EnginePool::Release(const std::string& modelId,
                           std::shared_ptr<IModelBackend> engine) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = pools_[modelId];
    // 池满时丢弃引擎实例（不阻塞，不等待）
    // 调用方可忽略此状态码，推理仍可正常进行
    if (entry.free.size() >= maxEngines_) {
        return Status{StatusCode::ErrorResourceExhaust, "Engine pool full"};
    }
    entry.free.push_back(std::move(engine));
    return Status{};
}

/**
 * 清空所有模型的引擎缓存
 * 所有缓存的 IModelBackend 实例将被销毁
 */
void EnginePool::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.clear();
}

} // namespace aicore
