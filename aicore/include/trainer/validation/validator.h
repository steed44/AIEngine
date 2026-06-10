#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <memory>
#include <vector>

namespace aicore {

struct ValidationResult {
    float map50 = 0;
    float map5095 = 0;
    float precision = 0;
    float recall = 0;
    int totalSamples = 0;
};

class Validator {
public:
    Validator();
    Status Validate(std::shared_ptr<IDataset> dataset,
                    std::function<std::vector<NodeResult>(const cv::Mat&)> inferFn,
                    ValidationResult& result);
    std::string GetLastError() const;

private:
    std::string lastError_;
};

} // namespace aicore
