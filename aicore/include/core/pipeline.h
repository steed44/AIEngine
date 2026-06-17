// pipeline（处理流水线）接口与状态定义
// 定义了一条完整的 AI 推理流水线的生命周期和行为
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
enum class PipelineState { kCreated, kReady, kRunning, kStopped, kError };

// pipeline 的抽象接口
// 负责管理处理器节点的拓扑结构、帧数据的流转调度以及生命周期控制
class IPipeline {
public:
    virtual ~IPipeline() = default;

    // 根据 JSON 配置字符串构建 pipeline 的拓扑结构
    // @param configJson 描述节点和边的 JSON 配置
    // @return 成功返回 Status::kOk
    virtual Status Build(const std::string& configJson) = 0;
    // 同步执行推理：输入一帧，等待处理完成后获取结果
    // @param input  输入帧
    // @param output 输出结果（引用传出）
    // @return 成功返回 Status::kOk
    virtual Status Execute(const Frame& input, Result& output) = 0;
    // 异步执行推理：输入一帧，通过回调接收结果
    // @param input    输入帧
    // @param callback 处理完成后的回调函数
    // @return 成功返回 Status::kOk
    virtual Status ExecuteAsync(const Frame& input,
                                std::function<void(const Result&)> callback) = 0;
    // 等待所有异步任务执行完毕
    // @return 成功返回 Status::kOk
    virtual Status WaitAll() = 0;
    // 停止 pipeline 的所有处理活动
    virtual void Stop() = 0;
    // 获取当前 pipeline 的状态
    // @return 当前状态值
    virtual PipelineState GetState() const = 0;
    // 获取当前 pipeline 的 JSON 配置字符串
    // @return 配置的 JSON 表示
    virtual std::string GetConfig() const = 0;
};

} // namespace aicore
