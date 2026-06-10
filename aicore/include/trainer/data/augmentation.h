#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <memory>
#include <vector>

namespace aicore {

class IAugmentation {
public:
    virtual ~IAugmentation() = default;
    virtual Sample Apply(const Sample& input) = 0;
};

class AugmentationPipeline {
public:
    void Add(std::shared_ptr<IAugmentation> aug);
    Sample Apply(const Sample& input);
    void Clear();

private:
    std::vector<std::shared_ptr<IAugmentation>> augs_;
};

} // namespace aicore
