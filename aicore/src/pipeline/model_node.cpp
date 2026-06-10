#include "pipeline/model_node.h"

namespace aicore {

ModelNode::ModelNode(std::shared_ptr<IModelBackend> backend)
    : backend_(std::move(backend)) {}

Status ModelNode::Init(const NodeConfig& config) {
    auto it = config.find("confidence_threshold");
    if (it != config.end())
        confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

Status ModelNode::Process(const std::vector<Frame>& inputs,
                          std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "no input frames"};

    for (const auto& frame : inputs) {
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

std::string ModelNode::GetName() const {
    return backend_ ? "model" : "model(uninitialized)";
}

std::string ModelNode::GetType() const {
    return "model";
}

} // namespace aicore
