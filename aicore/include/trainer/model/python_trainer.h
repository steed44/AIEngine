// Python 训练器，嵌入 Python 解释器执行训练逻辑
// 适用于需要复用 Python 训练脚本或 PyTorch 训练流程的场景
#pragma once
#include "core/types.h"
#include "optimizer/python_embedding.h"
#include <string>

namespace aicore {

// Python 训练器类
// 通过内嵌 Python 解释器，调用 Python 训练脚本完成模型训练
// 用于桥接 C++ 训练框架与现有 Python 训练生态
class PythonTrainer {
public:
    PythonTrainer();

    // 根据 JSON 配置启动 Python 训练
    // @param configJson  JSON 格式的训练参数（含脚本路径、超参数等）
    // @return 成功返回 Success，失败可通过 GetLastError 获取错误详情
    Status Train(const std::string& configJson);

    // 获取最近一次训练的错误信息
    // @return 错误描述字符串（无错误时返回空字符串）
    std::string GetLastError() const;

private:
    PythonEmbedding py_;       // Python 嵌入式解释器实例
    std::string lastError_;    // 最近一次错误的描述信息
};

} // namespace aicore
