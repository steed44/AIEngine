// ============================================================
// 文件：python_embedding.h
// 用途：定义 Python 运行时嵌入管理类，用于在 C++ 进程中
//   内嵌调用 Python 脚本（如 ONNX 导出、模型训练等）。
// ============================================================
#pragma once
#include "core/types.h"
#include <string>
#include <functional>

namespace aicore {

// PythonEmbedding：Python 解释器嵌入管理器
// 职责：初始化/销毁 Python 解释器，运行 Python 脚本
//   并传递 JSON 格式的配置参数、接收执行结果。
// 线程安全：RunScript 可在任意线程调用（自动获取 GIL）。
class AICORE_OPTIMIZER_API PythonEmbedding {
public:
    // 进度回调：接收 JSON 字符串（含 epoch/loss/metric 等字段）
    using ProgressCallback = std::function<void(const std::string& progressJson)>;

    PythonEmbedding();
    ~PythonEmbedding();

    // 初始化 Python 解释器（可重复调用，第二次起无操作）
    // 需在首次 RunScript 前调用，或由 RunScript 自动调用
    Status Initialize();

    // 运行指定的 Python 脚本
    // 参数 script     : Python 脚本文件路径（相对/绝对路径均可）
    // 参数 configJson : 以 JSON 字符串形式传递的配置参数
    // 参数 output     : 输出参数，接收脚本 train() 函数返回的 JSON
    // 原理：import module → train(configJson) → 返回值
    Status RunScript(const std::string& script, const std::string& configJson,
                     std::string& output);

    // 注册进度回调（RunScript 前调用）
    // Python 脚本调用 progress_hook(json_str) 时会触发此回调
    void SetProgressCallback(ProgressCallback cb) { progressCb_ = std::move(cb); }

    // 销毁 Python 解释器，释放资源
    void Finalize();

private:
    bool initialized_ = false;  // 标记 Python 解释器是否已初始化
    ProgressCallback progressCb_;  // 进度回调

    // 确保解释器已初始化
    Status ensureInitialized();
};

} // namespace aicore
