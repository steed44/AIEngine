// ============================================================
// 文件：model_optimizer.cpp
// 用途：实现模型优化器主类，编排 ONNX 导出与 TensorRT
//   引擎构建的完整优化流水线。
// ============================================================
#include "optimizer/model_optimizer.h"
#include <nlohmann/json.hpp>

namespace aicore {

ModelOptimizer::ModelOptimizer() {}

// Optimize 方法执行完整的模型优化流程：
// 1. 解析 JSON 配置字符串
// 2. 如果配置了 onnx_path，调用 OnnxExporter 导出 ONNX
// 3. 如果配置了 engine_path，调用 TensorRtBuilder 构建引擎
// 4. 配置解析异常时捕获并返回错误状态
Status ModelOptimizer::Optimize(const std::string& configJson) {
    try {
        // 使用 nlohmann/json 解析配置
        auto j = nlohmann::json::parse(configJson);
        std::string modelPath = j.value("model_path", "");
        std::string onnxPath = j.value("onnx_path", "");
        std::string enginePath = j.value("engine_path", "");
        std::string precision = j.value("precision", "fp16");

        // 必填字段校验
        if (modelPath.empty()) {
            lastError_ = "model_path is required";
            return Status{StatusCode::ErrorConfigParse, lastError_};
        }

        // 阶段一：ONNX 导出（可选）
        if (!onnxPath.empty()) {
            // 使用默认输入形状 {1,3,640,640} 导出 ONNX
            auto s = exporter_.Export(modelPath, onnxPath, {1, 3, 640, 640});
            if (!s) { lastError_ = "ONNX export failed: " + s.message; return s; }
            // 导出后执行简化（常量折叠、节点消除等）
            exporter_.Simplify(onnxPath);
        }

        // 阶段二：TensorRT 引擎构建（可选）
        if (!enginePath.empty()) {
            BuildConfig bc;
            bc.onnxPath = onnxPath;
            bc.enginePath = enginePath;
            // 根据配置字符串选择精度模式
            bc.precision = (precision == "int8") ? Precision::kINT8
                         : (precision == "fp16") ? Precision::kFP16
                         : Precision::kFP32;
            auto s = trtBuilder_.Build(bc);
            if (!s) { lastError_ = "TensorRT build failed: " + s.message; return s; }
        }

        return Status{};
    } catch (const std::exception& e) {
        // JSON 解析异常处理
        lastError_ = std::string("config parse error: ") + e.what();
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
}

std::string ModelOptimizer::GetLastError() const { return lastError_; }

} // namespace aicore
