// 模型推理节点 — 封装 IModelBackend 作为流水线处理步骤
//
// 工作流程：
//   输入帧 → 包装为 NCHW Tensor → backend_->Infer → 输出 Tensor
//  ，
// 后端抽象：
//   IModelBackend 接口统一 ONNX Runtime / TensorRT / LibTorch 三种后端。
//   具体实现在 BackendFactory::Create() 中按 backendType 创建。
#pragma once
#include "core/processor.h"
#include "core/model_backend.h"

namespace aicore {

// 模型推理节点
// 将输入帧转换为张量，调用后端推理引擎执行前向传播
class ModelNode : public IProcessor {
public:
    explicit ModelNode(std::unique_ptr<IModelBackend> backend);
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;
    // 获取底层的模型后端实例（可用于手动控制后端生命周期）
    std::shared_ptr<IModelBackend> GetBackend() const { return backend_; }

private:
    std::shared_ptr<IModelBackend> backend_;     // 模型后端（管理生命周期）
    float confidenceThreshold_ = 0.5f;           // 结果过滤置信度阈值
};

} // namespace aicore
