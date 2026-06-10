#pragma once
#include "trainer/data/dataset.h"
#include <vector>
#include <string>

namespace aicore {

class FolderDataset : public IDataset {
public:
    Status Load(const std::string& folderPath) override;
    size_t Size() const override;
    Sample Get(size_t index) override;
    int NumClasses() const override { return 1; }

private:
    std::vector<Sample> samples_;
};

} // namespace aicore
