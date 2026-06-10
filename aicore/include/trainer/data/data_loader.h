#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <vector>
#include <random>

namespace aicore {

struct Batch {
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    std::vector<BBox> bboxes;
};

class DataLoader {
public:
    DataLoader(std::shared_ptr<IDataset> dataset, int batchSize = 16, bool shuffle = true);
    Batch Next();
    bool HasNext() const;
    void Reset();
    size_t NumBatches() const;

private:
    std::shared_ptr<IDataset> dataset_;
    int batchSize_;
    bool shuffle_;
    size_t current_ = 0;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
};

} // namespace aicore
