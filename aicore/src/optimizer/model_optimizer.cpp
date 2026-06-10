#include "optimizer/model_optimizer.h"
#include <nlohmann/json.hpp>

namespace aicore {

ModelOptimizer::ModelOptimizer() {}

Status ModelOptimizer::Optimize(const std::string& configJson) {
    try {
        auto j = nlohmann::json::parse(configJson);
        std::string modelPath = j.value("model_path", "");
        std::string onnxPath = j.value("onnx_path", "");
        std::string enginePath = j.value("engine_path", "");
        std::string precision = j.value("precision", "fp16");

        if (modelPath.empty()) {
            lastError_ = "model_path is required";
            return Status{StatusCode::ErrorConfigParse, lastError_};
        }

        if (!onnxPath.empty()) {
            auto s = exporter_.Export(modelPath, onnxPath, {1, 3, 640, 640});
            if (!s) { lastError_ = "ONNX export failed: " + s.message; return s; }
            exporter_.Simplify(onnxPath);
        }

        if (!enginePath.empty()) {
            BuildConfig bc;
            bc.onnxPath = onnxPath;
            bc.enginePath = enginePath;
            bc.precision = (precision == "int8") ? Precision::kINT8
                         : (precision == "fp16") ? Precision::kFP16
                         : Precision::kFP32;
            auto s = trtBuilder_.Build(bc);
            if (!s) { lastError_ = "TensorRT build failed: " + s.message; return s; }
        }

        return Status{};
    } catch (const std::exception& e) {
        lastError_ = std::string("config parse error: ") + e.what();
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
}

std::string ModelOptimizer::GetLastError() const { return lastError_; }

} // namespace aicore
