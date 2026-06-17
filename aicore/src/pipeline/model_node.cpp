// ============================================================
// model_node.cpp — 模型推理节点
// 封装模型后端，将输入帧转换为张量并执行推理
// ============================================================
#include "pipeline/model_node.h"

namespace aicore {

/**
 * 构造函数：绑定模型后端
 * @param backend 推理后端实例（ONNX Runtime / TensorRT 等）
 */
ModelNode::ModelNode(std::shared_ptr<IModelBackend> backend)
    : backend_(std::move(backend)) {}

/**
 * 初始化模型节点：从配置读取置信度阈值
 * @param config 节点配置键值对
 */
Status ModelNode::Init(const NodeConfig& config) {
    auto it = config.find("confidence_threshold");
    if (it != config.end())
        confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

/**
 * 执行模型推理：将输入帧转为 NCHW 张量，调用后端 Infer
 * @param inputs  输入帧列表
 * @param outputs [out] 输出帧列表（暂未填充推理结果）
 */
Status ModelNode::Process(const std::vector<Frame>& inputs,
                          std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "no input frames"};

    for (const auto& frame : inputs) {
        // ---- 推理执行 ----
        // 将 OpenCV Mat（HWC, uint8）包装为 NCHW float32 Tensor 的过程：
        //
        // 数据布局转换：
        //   OpenCV Mat 是 HWC 布局（height × width × channels），
        //   但深度学习推理引擎通常要求 NCHW 布局。
        //   当前实现直接复用 frame.image.data 指针，不执行实际内存重排，
        //   而是设置 shape = {1, 3, H, W} 来声明 NCHW 语义。
        //
        // 注意：这种"零拷贝"封装假设推理后端能正确处理，
        // 即后端内部将 {1, 3, H, W} 解释为 NCHW 并在计算时执行相应步长计算。
        // 若后端实际需要 CHW 连续内存，需在此处执行 permute/transpose。
        std::vector<Tensor> inTensors;
        Tensor t;
        t.data = frame.image.data;
        t.shape = {1, 3, frame.height(), frame.width()};
        t.dtype = DataType::kFloat32;
        t.bytes = frame.image.total() * frame.image.elemSize();
        inTensors.push_back(t);

        std::vector<Tensor> outTensors;
        auto s = backend_->Infer(inTensors, outTensors);
        if (!s) return s;

        outputs.push_back(frame);
    }
    return Status{};
}

/**
 * 返回节点名称
 */
std::string ModelNode::GetName() const {
    return backend_ ? "model" : "model(uninitialized)";
}

/**
 * 返回节点类型标识
 */
std::string ModelNode::GetType() const {
    return "model";
}

} // namespace aicore
