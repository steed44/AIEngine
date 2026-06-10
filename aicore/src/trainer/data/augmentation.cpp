#include "trainer/data/augmentation.h"

namespace aicore {

void AugmentationPipeline::Add(std::shared_ptr<IAugmentation> aug) {
    augs_.push_back(std::move(aug));
}

Sample AugmentationPipeline::Apply(const Sample& input) {
    Sample out = input;
    for (auto& aug : augs_)
        out = aug->Apply(out);
    return out;
}

void AugmentationPipeline::Clear() { augs_.clear(); }

} // namespace aicore
