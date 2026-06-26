// 模型引擎池
// 管理和复用 IModelBackend 实例，避免重复加载模型，支持按模型 ID 的获取/释放
//
// 类似数据库连接池模式：
//   预创建多个推理引擎实例（每个 modelId 一个子池），
//   按需分配（Acquire）和归还（Release）。
//
// 为什么需要引擎池而非每次创建新引擎：
//   1. 模型加载开销大（TensorRT 引擎构建可能耗时数秒）
//   2. GPU 显存分配是昂贵操作
//   3. 避免反复创建/销毁 cuDNN/cuBLAS handle
//
// 当前局限性：
//   - 仅按 modelId 分组，不支持多模型热替换
//   - 池满时直接丢弃，无 LRU 淘汰
//   - 无自动扩缩容
#pragma once
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <memory>
#include <unordered_map>
#include <mutex>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 模型引擎池
// 线程安全的对象池，按 modelId 缓存 IModelBackend 实例
// 当池中空闲实例不足时自动创建新实例（不超过 maxEngines 限制）
class EnginePool {
public:
    // 构造引擎池
    // @param maxEngines 每个模型的最大缓存引擎数，默认 4
    explicit EnginePool(size_t maxEngines = 4);

    // 从池中获取一个模型后端实例
    // @param modelId 模型唯一标识
    // @param info    模型信息（首次加载时使用）
    // @param engine  输出的后端实例共享指针（引用传出）
    // @return 成功返回 Status::kOk
    Status Acquire(const std::string& modelId, const ModelInfo& info,
                   std::shared_ptr<IModelBackend>& engine);
    // 将模型后端实例归还到池中
    // @param modelId 模型唯一标识
    // @param engine  待归还的后端实例
    // @return 成功返回 Status::kOk
    Status Release(const std::string& modelId,
                   std::shared_ptr<IModelBackend> engine);
    // 清空所有缓存的后端实例
    void Clear();

private:
    // 池条目：包含某个 modelId 的所有空闲后端实例
    struct PoolEntry {
        std::vector<std::shared_ptr<IModelBackend>> free;   // 空闲引擎实例列表（后进先出）
        ModelInfo info;                                      // 模型元信息（路径/后端类型/设备）
    };

    size_t maxEngines_;                                     // 每模型最大缓存数（默认 4）
    std::unordered_map<std::string, PoolEntry> pools_;      // modelId → 池条目映射
    std::mutex mutex_;                                      // 线程安全互斥锁（保护 pools_）
};

} // namespace aicore
