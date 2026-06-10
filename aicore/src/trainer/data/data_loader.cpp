#include "trainer/data/data_loader.h"

namespace aicore {

DataLoader::DataLoader(std::shared_ptr<IDataset> dataset, int batchSize, bool shuffle)
    : dataset_(dataset), batchSize_(batchSize), shuffle_(shuffle), rng_(std::random_device{}()) {
    Reset();
}

void DataLoader::Reset() {
    current_ = 0;
    indices_.resize(dataset_->Size());
    for (size_t i = 0; i < indices_.size(); ++i) indices_[i] = i;
    if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
}

Batch DataLoader::Next() {
    Batch batch;
    for (int i = 0; i < batchSize_ && HasNext(); ++i, ++current_) {
        auto sample = dataset_->Get(indices_[current_]);
        batch.images.push_back(sample.image);
        batch.labels.push_back(sample.label);
        batch.bboxes.push_back(sample.bbox);
    }
    return batch;
}

bool DataLoader::HasNext() const { return current_ < indices_.size(); }
size_t DataLoader::NumBatches() const { return (indices_.size() + batchSize_ - 1) / batchSize_; }

} // namespace aicore
