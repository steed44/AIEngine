// ============================================================
// 文件：python_embedding.h
// 用途：定义 Python 运行时嵌入管理类，用于在 C++ 进程中
//   内嵌调用 Python 脚本（如 ONNX 导出、模型验证等）。
// ============================================================
#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

// PythonEmbedding：Python 解释器嵌入管理器
// 职责：初始化/销毁 Python 解释器，运行 Python 脚本
//   并传递 JSON 格式的配置参数、接收执行结果。
// 典型使用场景：在 TensorRT 优化流程中调用 PyTorch 的
//   torch.onnx.export() 完成模型格式转换。
class AICORE_OPTIMIZER_API PythonEmbedding {
public:
    PythonEmbedding();
    ~PythonEmbedding();

    // 初始化 Python 解释器
    Status Initialize();

    // 运行指定的 Python 脚本
    // 参数 script     : Python 脚本文件路径
    // 参数 configJson : 以 JSON 字符串形式传递的配置参数
    // 参数 output     : 输出参数，接收脚本执行返回的 JSON 结果
    Status RunScript(const std::string& script, const std::string& configJson,
                     std::string& output);

    // 销毁 Python 解释器，释放资源
    void Finalize();

private:
    bool initialized_ = false;  // 标记 Python 解释器是否已初始化
};

} // namespace aicore
