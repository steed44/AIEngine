#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace aicore {

struct Sample {
    cv::Mat image;
    int label = 0;
    std::string imagePath;
    BBox bbox;
};

class IDataset {
public:
    virtual ~IDataset() = default;
    virtual Status Load(const std::string& path) = 0;
    virtual size_t Size() const = 0;
    virtual Sample Get(size_t index) = 0;
    virtual int NumClasses() const = 0;
};

class COCODataset : public IDataset {
public:
    Status Load(const std::string& path) override;
    size_t Size() const override;
    Sample Get(size_t index) override;
    int NumClasses() const override;
private:
    std::vector<Sample> samples_;
};

} // namespace aicore
