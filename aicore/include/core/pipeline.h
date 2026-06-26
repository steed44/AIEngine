// pipeline（处理流水线）接口与状态定义
// 定义了一条完整的 AI 推理流水线的生命周期和行为
//
// 设计模式：模板方法模式（Template Method）
//   IPipeline 定义执行骨架（Build → Execute/ExecuteAsync → Stop），
//   子类 PipelineImpl 实现 DAG 编排细节
//
// 状态机：
//   kCreated → Build() → kReady → Execute() → kRunning → Stop() → kStopped
//                                    ↓ error
//                                  kError
//
// 线程安全：
//   Build()     — 非线程安全，构建阶段应单线程调用
//   Execute()   — 线程安全，支持多线程并发同步调用
//   ExecuteAsync— 线程安全，回调在内部线程池中执行
//   Stop()      — 线程安全，可在任意线程调用以终止运行
#pragma once
#include "core/types.h"
#include "core/frame.h"
#include "core/processor.h"
#include <memory>
#include <string>
#include <vector>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// pipeline 的生命周期状态枚举
//   kCreated:  已创建但未配置
//   kReady:    已构建完成，准备执行
//   kRunning:  正在处理帧数据
//   kStopped:  已停止处理
//   kError:    运行时错误状态
//
// 状态转换规则：
//   kCreated → Build() 成功 → kReady
//   kReady → Execute()/ExecuteAsync() → kRunning
//   kRunning → 所有任务完成 → kReady
//   kAny → Stop() → kStopped（终止态）
//   kAny → 内部错误 → kError（终止态，需重建 pipeline）
enum class PipelineState { kCreated, kReady, kRunning, kStopped, kError };

// pipeline 的抽象接口
// 负责管理处理器节点的拓扑结构、帧数据的流转调度以及生命周期控制
class IPipeline {
public:
    virtual ~IPipeline() = default;

    // 根据 JSON 配置字符串构建 pipeline 的拓扑结构
    // @param configJson 描述节点和边的 JSON 配置
    // @return 成功返回 Status::kOk
    // 前置条件：pipeline 处于 kCreated 状态
    // 后置条件：成功时状态变为 kReady，节点和边已实例化连接
    virtual Status Build(const std::string& configJson) = 0;
    // 同步执行推理：输入一帧，等待处理完成后获取结果
    // @param input  输入帧
    // @param output 输出结果（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：Build() 已成功调用，pipeline 处于 kReady 或 kRunning 状态
    // 后置条件：output 填充完整推理结果，pipeline 回到 kReady 状态
    virtual Status Execute(const Frame& input, Result& output) = 0;
    // 异步执行推理：输入一帧，通过回调接收结果
    // @param input    输入帧
    // @param callback 处理完成后的回调函数
    // @return 成功返回 Status::kOk
    // 前置条件：Build() 已成功调用
    // 后置条件：callback 会在 pipeline 内部线程池中被调用
    // 线程安全：可在任意线程发起异步调用
    virtual Status ExecuteAsync(const Frame& input,
                                std::function<void(const Result&)> callback) = 0;
    // 等待所有异步任务执行完毕
    // @return 成功返回 Status::kOk
    // 前置条件：至少一次 ExecuteAsync() 调用
    // 后置条件：所有已提交的异步任务均已回调完毕
    virtual Status WaitAll() = 0;
    // 停止 pipeline 的所有处理活动
    // 后置条件：状态变为 kStopped，未完成的异步任务被取消
    // 幂等性：重复调用无额外效果
    virtual void Stop() = 0;
    // 获取当前 pipeline 的状态
    // @return 当前状态值
    // 线程安全：const 方法，线程安全
    virtual PipelineState GetState() const = 0;
    // 获取当前 pipeline 的 JSON 配置字符串
    // @return 配置的 JSON 表示
    // 线程安全：const 方法，线程安全
    virtual std::string GetConfig() const = 0;
};

} // namespace aicore
