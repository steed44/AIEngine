// 处理器节点抽象接口
// pipeline 中的每个处理步骤（预处理、推理、后处理等）均由 IProcessor 实现
//
// 设计模式：策略模式（Strategy）
//   IProcessor 是策略接口，每种处理逻辑（resize/normalize/model/nms 等）
//   各自实现此接口，PipelineBuilder 根据配置动态选择并组装策略对象
//
// 线程安全：
//   Init()   — 非线程安全，应在 pipeline 启动前单线程调用
//   Process()— 实现应支持多线程并发调用（pipeline 内部通过 ThreadPool 并行）
//   GetName/GetType — 只读操作，线程安全
//
// 生命周期：
//   1. 构造 → 2. Init(config) → 3. Process(inputs→outputs) 可重复 → 4. 析构
#pragma once
#include "core/types.h"
#include "core/frame.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace aicore { class ThreadPool; }

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 处理器节点的配置字典：键值对形式，由具体处理器子类定义支持的参数
// 设计说明：使用字符串字典而非强类型结构体，允许每个子类灵活定义自己的配置项，
//           无需为每个节点类型单独定义配置结构。配置项由 Init() 内部解析验证
using NodeConfig = std::unordered_map<std::string, std::string>;

// 处理器节点的抽象基类
// 所有具体的处理逻辑（如缩放、归一化、模型推理、解码等）都实现此接口
class IProcessor {
public:
    virtual ~IProcessor() = default;

    // 初始化处理器，传入节点级别的配置参数
    // @param config 配置键值对字典
    // @return 成功返回 Status::kOk
    // 前置条件：构造函数已执行完毕，config 包含该节点所需全部参数
    // 后置条件：处理器进入就绪状态，可被 Process() 调用
    // 设计说明：Init 与构造函数分离，使得对象可被工厂统一构造后再配置
    virtual Status Init(const NodeConfig& config) = 0;
    // 处理输入的帧列表，生成输出帧列表
    // @param inputs  输入帧向量（可能包含多个输入源）
    // @param outputs 输出帧向量（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：Init() 已成功调用
    // 后置条件：outputs 包含处理结果帧；inputs 保持不变
    // 线程安全：允许多线程并发调用不同实例的 Process()，或同一实例处理不同 inputs
    virtual Status Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) = 0;
    // 获取处理器名称标识
    // @return 处理器名称字符串
    // 线程安全：const 方法，线程安全
    virtual std::string GetName() const = 0;
    // 获取处理器类型标识（如 "Preprocessor", "Inference", "Postprocessor" 等）
    // @return 处理器类型字符串
    // 线程安全：const 方法，线程安全
    virtual std::string GetType() const = 0;

    // 注入线程池引用，支持并行推理的节点可覆写此方法
    // 设计说明：pipeline 内部线程池通过此方法注入，节点内可提交并行任务，
    //           默认空实现确保不需要并行的节点无需关注此接口
    // @param pool 线程池指针，可为 nullptr（表示不使用线程池）
    virtual void SetThreadPool(ThreadPool* pool) { (void)pool; }
};

} // namespace aicore
