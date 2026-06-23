#pragma once
#include <opencv2/core.hpp>
#include <torch/nn.h>
#include <string>
#include <vector>
#include <memory>
#include <random>

namespace aicore {

struct YOLOSample {
    cv::Mat image;
    std::string imagePath;
    std::vector<cv::Rect2f> boxes; // [N] (cx,cy,w,h) 归一化
    std::vector<int> labels;
};

struct YOLOBatch {
    torch::Tensor images;  // [B, 3, H, W] uint8, RGB
    torch::Tensor targets; // [M, 6] (batchIdx, cls, cx, cy, w, h) 归一化
    int numClasses = 0;
};

class YOLODataset {
public:
    bool Load(const std::string& imgDir, const std::string& labelDir = "");
    size_t Size() const { return imgPaths_.size(); }
    YOLOSample Get(size_t idx);
    int NumClasses() const { return numClasses_; }
    std::string GetPath(size_t idx) const { return imgPaths_[idx]; }

private:
    bool loadLabels(const std::string& labelPath, std::vector<float>& out);
    std::vector<std::string> imgPaths_;
    std::vector<std::string> labelFiles_;
    int numClasses_ = 0;
};

// ─── 增强函数 ────────────────────────────────────────────────

// Mosaic: 4 图拼接成 1 张 (outSize×outSize), 调整标签坐标
YOLOSample mosaicAugment(
    const std::vector<YOLOSample>& samples,
    int outSize = 640,
    int centerRange = 0);  // center jitter range (0=全图随机)

// HSV Jitter: 随机调整 H/S/V (YOLOv8 默认 h=5, s=30, v=30)
YOLOSample hsvJitter(const YOLOSample& sample, float hGain = 5, float sGain = 30, float vGain = 30);

// 随机水平翻转 + box 坐标适配
YOLOSample randomFlipH(const YOLOSample& sample, float prob = 0.5);

// 归一化缩放到 targetSize, 等比例填充灰边
YOLOSample letterbox(const YOLOSample& sample, int targetSize = 640);

// ─── DataLoader ────────────────────────────────────────────────

class YOLODataLoader {
public:
    YOLODataLoader(std::shared_ptr<YOLODataset> dataset, int batchSize = 16,
                   int imgSize = 640, bool shuffle = true);

    YOLOBatch Next();
    bool HasNext() const;
    void Reset();
    size_t NumBatches() const;

    bool mosaic = true;
    bool hsvJitterOn = true;
    bool flip = true;

private:
    std::vector<YOLOSample> loadBatch();
    YOLOBatch collate(const std::vector<YOLOSample>& samples);

    std::shared_ptr<YOLODataset> dataset_;
    int batchSize_;
    int imgSize_;
    bool shuffle_;
    size_t current_ = 0;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
};

} // namespace aicore
