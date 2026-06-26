// AI 引擎单例 — 顶层推理入口
// 管理 pipeline 生命周期、线程安全执行和资源池
//
// 设计模式：单例（Meyer's Singleton）
//   - GetInstance() 返回 static 局部变量，C++11 保证线程安全初始化
//
// 线程安全：
//   - Init/Execute/Shutdown 均持有 mutex_ 互斥锁
//   - 所有公开方法互斥，同一时刻仅允许一个操作
//
// 资源生命周期：
//   pipeline_   ← unique_ptr，随 Init/Shutdown 创建销毁
//   enginePool_ ← shared_ptr，可被 PipelineImpl 共享持有
#pragma once
#include "core/pipeline.h"
#include "core/frame.h"
#include "engine/engine_pool.h"
#include <memory>
#include <mutex>

namespace aicore {

// 引擎单例类
// 提供 Init/Execute/Shutdown 全生命周期管理，全局唯一实例
class AiEngine {
public:
    // 获取全局单例实例
    static AiEngine& GetInstance();

    // 根据 JSON 配置初始化引擎（解析配置 -> 构建 pipeline -> 准备引擎池）
    Status Init(const std::string& configJson);
    // 同步执行推理，线程安全
    Status Execute(const Frame& input, Result& output);
    // 异步执行推理，通过回调接收结果
    Status ExecuteAsync(const Frame& input,
                        std::function<void(const Result&)> callback);
    // 关闭引擎，释放所有资源
    void Shutdown();
    // 查询当前 pipeline 状态
    PipelineState GetState() const;

private:
    AiEngine() = default;
    ~AiEngine();
    AiEngine(const AiEngine&) = delete;
    AiEngine& operator=(const AiEngine&) = delete;

    std::unique_ptr<IPipeline> pipeline_;       // 推理流水线
    std::shared_ptr<EnginePool> enginePool_;    // 模型引擎池
    mutable std::mutex mutex_;                  // 线程安全互斥锁
};

} // namespace aicore
