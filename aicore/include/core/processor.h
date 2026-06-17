// 处理器节点抽象接口
// pipeline 中的每个处理步骤（预处理、推理、后处理等）均由 IProcessor 实现
#pragma once
#include "core/types.h"
#include "core/frame.h"
#include <string>
#include <unordered_map>
#include <memory>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 处理器节点的配置字典：键值对形式，由具体处理器子类定义支持的参数
using NodeConfig = std::unordered_map<std::string, std::string>;

// 处理器节点的抽象基类
// 所有具体的处理逻辑（如缩放、归一化、模型推理、解码等）都实现此接口
class IProcessor {
public:
    virtual ~IProcessor() = default;

    // 初始化处理器，传入节点级别的配置参数
    // @param config 配置键值对字典
    // @return 成功返回 Status::kOk
    virtual Status Init(const NodeConfig& config) = 0;
    // 处理输入的帧列表，生成输出帧列表
    // @param inputs  输入帧向量（可能包含多个输入源）
    // @param outputs 输出帧向量（引用传出）
    // @return 成功返回 Status::kOk
    virtual Status Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) = 0;
    // 获取处理器名称标识
    // @return 处理器名称字符串
    virtual std::string GetName() const = 0;
    // 获取处理器类型标识（如 "Preprocessor", "Inference", "Postprocessor" 等）
    // @return 处理器类型字符串
    virtual std::string GetType() const = 0;
};

} // namespace aicore
