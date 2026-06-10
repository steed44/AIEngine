#include "optimizer/tensorrt_builder.h"

namespace aicore {

TensorRtBuilder::TensorRtBuilder() {}
TensorRtBuilder::~TensorRtBuilder() {}

Status TensorRtBuilder::Build(const BuildConfig& config) {
    (void)config;
    return Status{StatusCode::ErrorInternal, "TensorRT build not available in stub"};
}

std::string TensorRtBuilder::GetLastError() const { return lastError_; }

} // namespace aicore
