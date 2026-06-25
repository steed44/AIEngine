// ============================================================
// onnxruntime_backend.cpp — ONNX Runtime 后端实现
//
// 本文件实现了 IModelBackend 接口的 ONNX Runtime 后端。
// 通过条件编译 AICORE_USE_ONNXRUNTIME 控制是否包含此实现。
//
// 内存管理说明（重要）：
//   输出 Tensor 的 data 指向新分配的 float[] 缓冲区，
//   allocId = 1 标记为 backend 分配，调用方需 delete[] 释放。
//   原因：ORT 的 GetTensorData<float>() 返回指向 session 内部缓冲区的指针，
//   如果直接返回，session 销毁后 data 成为悬垂指针。
//   因此必须用 memcpy 拷贝到独立缓冲区。
// ============================================================

#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <vector>
#include <string>
#include <cstring>

#ifdef AICORE_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace aicore {

/**
 * ONNX Runtime 后端 — 加载 .onnx 模型并执行推理
 *
 * 工作流程：
 *   Load():
 *     1. 创建 Ort::Env（ONNX Runtime 全局环境）
 *     2. 配置 SessionOptions：线程数=4，图优化级别=ALL
 *     3. 如果 deviceId >= 0，启用 CUDA 执行提供者
 *     4. 创建 Ort::Session 加载模型文件
 *     5. 遍历输入输出节点，缓存名称和形状
 *
 *   Infer():
 *     1. 将外部 Tensor 包装为 Ort::Value（CPU 内存）
 *     2. session_->Run() 执行推理
 *     3. 将结果从 Ort::Value 拷贝到独立缓冲区
 *     4. 填充 Tensor 结构并返回
 *
 * 性能调参：
 *   - IntraOpNumThreads=4：限制每个 session 使用的线程数
 *   - ORT_ENABLE_ALL：启用所有图优化（常量折叠、算子融合等）
 *   - CUDA Provider：启用 GPU 推理（如果配置了 deviceId）
 */
class ONNXRuntimeBackend : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }

    /**
     * 加载 ONNX 模型
     * @param info 模型信息：modelPath, deviceId
     */
    Status Load(const ModelInfo& info) override {
        modelPath_ = info.modelPath;
        deviceId_ = info.deviceId;
#ifdef AICORE_USE_ONNXRUNTIME
        try {
            // 创建运行环境，日志级别为 WARNING
            env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "aicore-ort");

            // 配置会话选项
            Ort::SessionOptions sessionOpts;
            sessionOpts.SetIntraOpNumThreads(4);  // 限制线程数，避免过度消耗 CPU
            sessionOpts.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);  // 启用所有图优化

            // 如果指定了 GPU 设备，启用 CUDA 执行提供者
            if (deviceId_ >= 0) {
                OrtCUDAProviderOptions cudaOpts;
                cudaOpts.device_id = deviceId_;
                sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
            }

            // 加载模型文件
            session_ = std::make_shared<Ort::Session>(*env_, modelPath_.c_str(), sessionOpts);

            // 缓存输入输出节点信息
            Ort::AllocatorWithDefaultOptions allocator;
            inputNames_.clear();
            outputNames_.clear();

            size_t numInputs = session_->GetInputCount();
            size_t numOutputs = session_->GetOutputCount();

            for (size_t i = 0; i < numInputs; i++) {
                auto name = session_->GetInputNameAllocated(i, allocator);
                inputNames_.push_back(name.get());
                auto typeInfo = session_->GetInputTypeInfo(i);
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                inputShapes_.push_back(shapeInfo.GetShape());
            }

            for (size_t i = 0; i < numOutputs; i++) {
                auto name = session_->GetOutputNameAllocated(i, allocator);
                outputNames_.push_back(name.get());
                auto typeInfo = session_->GetOutputTypeInfo(i);
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                outputShapes_.push_back(shapeInfo.GetShape());
            }

            loaded_ = true;
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelLoad,
                "ONNX Runtime load failed: " + std::string(e.what())};
        }
#else
        (void)info;
        return Status{StatusCode::ErrorModelLoad,
            "ONNX Runtime not available (recompile with AICORE_USE_ONNXRUNTIME)"};
#endif
    }

    /**
     * 执行 ONNX 模型推理
     * @param inputs 输入张量（CPU 内存，float32）
     * @param outputs 输出张量（调用方负责释放 data 内存）
     */
    Status Infer(const std::vector<Tensor>& inputs,
                 std::vector<Tensor>& outputs) override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_ || !session_) {
            return Status{StatusCode::ErrorModelLoad, "ONNX Runtime not loaded"};
        }
        try {
            // 步骤1: 将外部 Tensor 包装为 Ort::Value
            // 注意：Ort::Value::CreateTensor 不拷贝数据，直接引用外部内存
            std::vector<Ort::Value> ortInputs;
            std::vector<const char*> ortInputNames;
            std::vector<const char*> ortOutputNames;

            for (size_t i = 0; i < inputs.size() && i < inputNames_.size(); i++) {
                auto& t = inputs[i];
                std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
                Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
                    OrtArenaAllocator, OrtMemTypeDefault);
                ortInputs.push_back(Ort::Value::CreateTensor<float>(
                    memInfo, static_cast<float*>(t.data), t.bytes / sizeof(float),
                    shape.data(), shape.size()));
                ortInputNames.push_back(inputNames_[i].c_str());
            }

            for (auto& name : outputNames_) {
                ortOutputNames.push_back(name.c_str());
            }

            // 步骤2: 执行推理
            auto ortOutputs = session_->Run(
                Ort::RunOptions{nullptr},
                ortInputNames.data(), ortInputs.data(), ortInputs.size(),
                ortOutputNames.data(), ortOutputNames.size());

            // 步骤3: 将结果从 Ort::Value 拷贝到独立缓冲区
            // 必须拷贝！因为 ortOut.GetTensorData<float>() 指向 session 内部缓冲区，
            // session 可能被销毁或重用，导致悬垂指针。
            outputs.clear();
            outputs.reserve(ortOutputs.size());
            for (auto& ortOut : ortOutputs) {
                auto typeInfo = ortOut.GetTensorTypeAndShapeInfo();
                auto shape = typeInfo.GetShape();
                auto count = typeInfo.GetElementCount();

                Tensor t;
                t.dtype = DataType::kFloat32;
                t.shape.assign(shape.begin(), shape.end());
                t.bytes = count * sizeof(float);
                t.memory = MemoryType::kCPU;
                // 分配新缓冲区并拷贝数据
                auto* buf = new float[count];
                std::memcpy(buf, ortOut.GetTensorData<float>(), t.bytes);
                t.data = buf;
                t.allocId = 1;  // 标记 backend 分配，调用方需 delete[]
                outputs.push_back(t);
            }

            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelInfer,
                "ONNX Runtime infer failed: " + std::string(e.what())};
        }
#else
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal,
            "ONNX Runtime not available (recompile with AICORE_USE_ONNXRUNTIME)"};
#endif
    }

    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_) return Status{StatusCode::ErrorModelLoad, "not loaded"};
        shapes = inputShapes_;
        return Status{};
#else
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
#endif
    }

    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_) return Status{StatusCode::ErrorModelLoad, "not loaded"};
        shapes = outputShapes_;
        return Status{};
#else
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
#endif
    }

    void SetDeviceId(int deviceId) override { deviceId_ = deviceId; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }

private:
    std::string modelPath_;
    int deviceId_ = 0;
    bool loaded_ = false;
#ifdef AICORE_USE_ONNXRUNTIME
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::vector<std::vector<int64_t>> outputShapes_;
#endif
};

/**
 * 创建 ONNX Runtime 后端实例
 * 由 backend_factory.cpp 在条件编译时调用
 */
std::unique_ptr<IModelBackend> CreateONNXRuntimeBackend() {
    return std::make_unique<ONNXRuntimeBackend>();
}

} // namespace aicore
