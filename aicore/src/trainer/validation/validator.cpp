#include "trainer/validation/validator.h"

namespace aicore {

Validator::Validator() {}

Status Validator::Validate(std::shared_ptr<IDataset> dataset,
                            std::function<std::vector<NodeResult>(const cv::Mat&)> inferFn,
                            ValidationResult& result) {
    (void)dataset; (void)inferFn;
    result.map50 = 0;
    result.map5095 = 0;
    result.totalSamples = static_cast<int>(dataset ? dataset->Size() : 0);
    return Status{};
}

std::string Validator::GetLastError() const { return lastError_; }

} // namespace aicore
