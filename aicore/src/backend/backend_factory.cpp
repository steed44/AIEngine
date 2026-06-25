// ============================================================
// backend_factory.cpp — 后端引擎工厂
//
// 本文件实现了 IModelBackend 的后端工厂模式，根据 BackendType
// 枚举值创建对应的推理后端实例。支持三种后端：
//   - TensorRT:   NVIDIA TensorRT 引擎（stub，待实现）
//   - ONNXRuntime: 微软 ONNX Runtime（条件编译，见 onnxruntime_backend.cpp）
//   - LibTorch:   PyTorch LibTorch（完整实现，内联于此文件）
//
// 设计决策：
//   1. LibTorch 后端内联在此文件而非单独 .cpp 文件，
//      原因是跨翻译单元的链接问题：libtorch_backend.cpp 编译进 aicore.dll，
//      但 backend_factory.cpp 需要调用 CreateLibTorchBackend()，
//      静态链接会导致符号不可见。内联后所有代码在同一 TU 中，无链接问题。
//   2. 未启用 ONNX Runtime 或 LibTorch 时，使用 Stub 类返回
//      明确的错误信息，而非静默崩溃。
// ============================================================

#include "backend/backend_factory.h"

#ifdef AICORE_HAS_LIBTORCH
// LibTorch 头文件：torch::jit::script::Module 用于加载 .pt 模型
#include <torch/torch.h>
#include <torch/script.h>
#endif

#ifdef AICORE_USE_ONNXRUNTIME
// 内联 ONNX Runtime 后端实现（避免重复编译）
#include "backend/onnxruntime_backend.cpp"
#endif

namespace aicore {

// ============================================================
// LibTorch 后端实现（条件编译）
// ============================================================
#ifdef AICORE_HAS_LIBTORCH
/**
 * LibTorch 后端 — 加载 TorchScript 模型并执行推理
 *
 * 工作流程：
 *   Load():
 *     1. 根据 deviceId 选择 CPU 或 CUDA 设备
 *     2. torch::jit::load() 加载 .pt 序列化模型
 *     3. module_->to(device) 迁移到目标设备
 *     4. module_->eval() 关闭 dropout 等训练模式
 *
 *   Infer():
 *     1. 将 Tensor[] → torch::from_blob() 包装为 IValue
 *     2. module_->forward(ivInputs) 执行前向传播
 *     3. 结果可能是单个 Tensor 或 Tuple<IValue>
 *     4. 将结果拷贝到 CPU 内存（避免悬垂指针）
 *
 * 内存管理：
 *   输出 Tensor 的 data 指向新分配的 float[] 缓冲区，
 *   allocId = 1 标记为 backend 分配，调用方需 delete[] 释放。
 *   这样做的原因是：LibTorch 的 cpuTensor 是局部变量，
 *   函数返回后会被销毁，data_ptr<float>() 成为悬垂指针。
 */
class LibTorchBackend : public IModelBackend {
public:
    // 返回后端类型标识
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }

    /**
     * 加载 TorchScript 模型
     * @param info 模型信息：modelPath, deviceId, numInputs/Outputs
     */
    Status Load(const ModelInfo& info) override {
        modelPath_ = info.modelPath;
        deviceId_ = info.deviceId;
        try {
            // 自动选择设备：有 GPU 且指定了 deviceId 则用 CUDA，否则用 CPU
            torch::DeviceType deviceType = torch::kCPU;
            if (deviceId_ >= 0 && torch::cuda::is_available()) {
                deviceType = torch::kCUDA;
            }
            // torch::jit::load 加载 .pt 序列化模型，返回 script::Module
            module_ = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(modelPath_, deviceType));
            module_->to(deviceType);  // 迁移到目标设备
            module_->eval();          // 切换到推理模式（关闭 dropout 等）
            loaded_ = true;
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelLoad,
                std::string("LibTorch load failed: ") + e.what()};
        }
    }

    /**
     * 执行模型推理
     * @param inputs 输入张量列表（CPU 内存，float32）
     * @param outputs 输出张量列表（调用方负责释放 data 内存）
     */
    Status Infer(const std::vector<Tensor>& inputs,
                 std::vector<Tensor>& outputs) override {
        if (!loaded_ || !module_) {
            return Status{StatusCode::ErrorModelLoad, "LibTorch model not loaded"};
        }
        try {
            // 步骤1: 将外部 Tensor 包装为 LibTorch IValue
            // torch::from_blob 不拷贝数据，直接引用外部内存
            std::vector<torch::jit::IValue> ivInputs;
            for (auto& t : inputs) {
                auto tensor = torch::from_blob(
                    t.data,
                    torch::IntArrayRef(t.shape.data(), t.shape.size()),
                    torch::TensorOptions(torch::kFloat32));
                ivInputs.push_back(tensor);
            }

            // 步骤2: 执行前向传播
            // module_->forward() 返回 IValue，可能是 Tensor 或 Tuple<IValue>
            auto results = module_->forward(ivInputs);

            // 步骤3: 将结果从 GPU 拷贝到 CPU 并提取数据
            // 注意：必须拷贝！因为 cpuTensor 是局部变量，函数返回后销毁，
            // data_ptr<float>() 会成为悬垂指针。
            if (results.isTensor()) {
                auto out = results.toTensor();
                // contiguous() 确保内存连续，cpu() 迁移到 CPU
                auto cpuTensor = out.contiguous().cpu();
                auto count = cpuTensor.numel();
                Tensor t;
                t.dtype = DataType::kFloat32;
                t.shape.assign(cpuTensor.sizes().begin(), cpuTensor.sizes().end());
                t.bytes = count * sizeof(float);
                t.memory = MemoryType::kCPU;
                // 分配新缓冲区并拷贝数据
                auto* buf = new float[count];
                std::memcpy(buf, cpuTensor.data_ptr<float>(), t.bytes);
                t.data = buf;
                t.allocId = 1;  // 标记 backend 分配
                outputs.push_back(t);
            } else if (results.isTuple()) {
                // script::Module::forward 返回 Tuple<IValue> 时，
                // 遍历每个元素，提取 Tensor
                for (auto& elem : results.toTuple()->elements()) {
                    if (elem.isTensor()) {
                        auto out = elem.toTensor();
                        auto cpuTensor = out.contiguous().cpu();
                        auto count = cpuTensor.numel();
                        Tensor t;
                        t.dtype = DataType::kFloat32;
                        t.shape.assign(cpuTensor.sizes().begin(), cpuTensor.sizes().end());
                        t.bytes = count * sizeof(float);
                        t.memory = MemoryType::kCPU;
                        auto* buf = new float[count];
                        std::memcpy(buf, cpuTensor.data_ptr<float>(), t.bytes);
                        t.data = buf;
                        t.allocId = 1;
                        outputs.push_back(t);
                    }
                }
            }
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelInfer,
                std::string("LibTorch infer failed: ") + e.what()};
        }
    }

    // 形状查询暂不支持（TorchScript 模块无内置形状 introspection）
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "shape introspection not available"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "shape introspection not available"};
    }

    void SetDeviceId(int deviceId) override { deviceId_ = deviceId; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }

private:
    std::string modelPath_;
    int deviceId_ = 0;
    bool loaded_ = false;
    // 使用 unique_ptr 管理 Module 生命周期，避免 shared_ptr 的额外开销
    std::unique_ptr<torch::jit::script::Module> module_;
};
#endif

// ============================================================
// TensorRT 后端桩（Stub）
// ============================================================
/**
 * TensorRT 后端 — 当前为 Stub 实现
 *
 * Load() 返回成功（设置 loaded_=true），但 Infer() 返回错误。
 * 这样设计是为了让上层测试和调用方能正常创建后端实例，
 * 同时在使用时得到明确的"未实现"错误提示。
 */
class TensorRTBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kTensorRT; }
    Status Load(const ModelInfo& info) override {
        (void)info;
        loaded_ = true;
        return Status{};
    }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorModelInfer, "TensorRT backend not yet implemented"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "TensorRT backend not yet implemented"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "TensorRT backend not yet implemented"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

// ============================================================
// ONNX Runtime 后端桩（Stub）
// ============================================================
/**
 * ONNX Runtime 后端 — 当 AICORE_USE_ONNXRUNTIME 未定义时的 Stub
 *
 * 与 TensorRTBackendStub 类似：Load() 返回成功，Infer() 返回错误。
 * 实际实现见 onnxruntime_backend.cpp（条件编译时内联）。
 */
class ONNXRuntimeBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }
    Status Load(const ModelInfo& info) override {
        (void)info;
        loaded_ = true;
        return Status{};
    }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

// ============================================================
// LibTorch 后端桩（Stub）
// ============================================================
/**
 * LibTorch 后端 — 当 AICORE_HAS_LIBTORCH 未定义时的 Stub
 */
class LibTorchBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }
    Status Load(const ModelInfo& info) override {
        (void)info;
        loaded_ = true;
        return Status{};
    }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

// ============================================================
// 后端工厂 — 根据 BackendType 创建对应实例
// ============================================================
/**
 * 工厂方法：根据后端类型枚举创建 IModelBackend 实例
 *
 * 条件编译策略：
 *   - AICORE_HAS_LIBTORCH 定义 → 创建真实 LibTorchBackend
 *   - 未定义 → 创建 LibTorchBackendStub（返回错误）
 *   - AICORE_USE_ONNXRUNTIME 定义 → 调用 CreateONNXRuntimeBackend()
 *   - 未定义 → 创建 ONNXRuntimeBackendStub
 *   - TensorRT → 始终创建 TensorRTBackendStub（待实现）
 *
 * @param type 后端类型枚举
 * @return IModelBackend 唯一指针，未知类型返回 nullptr
 */
std::unique_ptr<IModelBackend> BackendFactory::Create(BackendType type) {
    switch (type) {
    case BackendType::kTensorRT:
        return std::make_unique<TensorRTBackendStub>();
    case BackendType::kONNXRuntime:
#ifdef AICORE_USE_ONNXRUNTIME
        return CreateONNXRuntimeBackend();
#else
        return std::make_unique<ONNXRuntimeBackendStub>();
#endif
    case BackendType::kLibTorch:
#ifdef AICORE_HAS_LIBTORCH
        return std::make_unique<LibTorchBackend>();
#else
        return std::make_unique<LibTorchBackendStub>();
#endif
    default:
        return nullptr;
    }
}

} // namespace aicore
