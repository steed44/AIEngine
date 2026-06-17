// ============================================================================
// 文件：python_trainer.cpp
// 用途：Python 训练器实现，通过嵌入的 Python 解释器调用外部训练脚本
// 功能：将训练配置传递给 Python 脚本并执行，支持 YOLO 等基于 Python 的训练
// ============================================================================

#include "trainer/model/python_trainer.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
PythonTrainer::PythonTrainer() {}

// 通过嵌入的 Python 解释器执行训练脚本
// 参数 configJson - JSON 格式的训练配置字符串（包含模型、数据、超参数等）
// 返回值         - 操作状态，成功为 Status{}，失败带回错误码
Status PythonTrainer::Train(const std::string& configJson) {
    std::string output;
    // 调用 Python 解释器运行 scripts/train_yolo.py，传入配置字符串
    return py_.RunScript("scripts/train_yolo.py", configJson, output);
}

// 获取最近一次训练操作的错误描述
// 返回值 - 错误信息字符串（无错误时返回空字符串）
std::string PythonTrainer::GetLastError() const { return lastError_; }

} // namespace aicore
