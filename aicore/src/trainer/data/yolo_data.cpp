#include "trainer/data/yolo_data.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace aicore {

namespace fs = std::filesystem;

// ============================================================
// YOLODataset
// ============================================================

bool YOLODataset::Load(const std::string& imgDir, const std::string& labelDir) {
    auto labelRoot = labelDir.empty() ? imgDir : labelDir;

    if (!fs::exists(imgDir) || !fs::exists(labelRoot)) return false;

    for (auto& entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        // 支持常见图片格式
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

        auto stem = entry.path().stem().string();
        auto labelPath = (fs::path(labelRoot) / (stem + ".txt")).string();

        if (!fs::exists(labelPath)) continue;

        std::vector<float> parsed;
        if (!loadLabels(labelPath, parsed)) continue;

        if (parsed.empty()) continue;  // 跳过无标签图片

        imgPaths_.push_back(entry.path().string());
        labelFiles_.push_back(labelPath);

        // 更新最大类别数
        for (size_t i = 0; i < parsed.size(); i += 5) {
            int cls = (int)parsed[i];
            if (cls + 1 > numClasses_) numClasses_ = cls + 1;
        }
    }
    return !imgPaths_.empty();
}

bool YOLODataset::loadLabels(const std::string& labelPath, std::vector<float>& out) {
    std::ifstream f(labelPath);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        float cls, cx, cy, w, h;
        if (!(ss >> cls >> cx >> cy >> w >> h)) continue;
        // 过滤无效标签
        if (w <= 0 || h <= 0 || cx < 0 || cx > 1 || cy < 0 || cy > 1) continue;
        out.push_back(cls);
        out.push_back(cx);
        out.push_back(cy);
        out.push_back(w);
        out.push_back(h);
    }
    return !out.empty();
}

YOLOSample YOLODataset::Get(size_t idx) {
    YOLOSample s;
    if (idx >= imgPaths_.size()) return s;

    s.imagePath = imgPaths_[idx];
    s.image = cv::imread(s.imagePath);
    if (s.image.empty()) return s;

    // 从 labelFiles_ 重新解析标签 (避免存储巨大内存)
    std::vector<float> parsed;
    loadLabels(labelFiles_[idx], parsed);
    for (size_t i = 0; i < parsed.size(); i += 5) {
        s.labels.push_back((int)parsed[i]);
        s.boxes.push_back(cv::Rect2f(parsed[i+1], parsed[i+2], parsed[i+3], parsed[i+4]));
    }
    return s;
}

// ============================================================
// Mosaic: 4 图 → 1 张
// ============================================================
YOLOSample mosaicAugment(const std::vector<YOLOSample>& samples, int outSize, int centerRange) {
    YOLOSample result;
    if (samples.size() < 4) return samples.empty() ? YOLOSample{} : samples[0];

    std::mt19937 rng(std::random_device{}());
    using uni = std::uniform_int_distribution<int>;
    int cx = centerRange > 0 ? uni(centerRange, outSize - centerRange)(rng)
                             : uni(outSize / 3, 2 * outSize / 3)(rng);
    int cy = centerRange > 0 ? uni(centerRange, outSize - centerRange)(rng)
                             : uni(outSize / 3, 2 * outSize / 3)(rng);

    // 拼接画布
    result.image = cv::Mat(outSize, outSize, CV_8UC3, cv::Scalar(114, 114, 114));
    auto box = [&](int i, int& xOff, int& yOff, float& scale) {
        auto& img = samples[i].image;
        scale = std::uniform_real_distribution(0.5f, 1.5f)(rng);
        int newW = int(img.cols * scale);
        int newH = int(img.rows * scale);
        scale = (float)newW / img.cols; // actual scale

        switch (i) {
            case 0: xOff = cx - newW; yOff = cy - newH; break; // top-left
            case 1: xOff = cx;        yOff = cy - newH; break; // top-right
            case 2: xOff = cx - newW; yOff = cy;        break; // bottom-left
            case 3: xOff = cx;        yOff = cy;        break; // bottom-right
        }
        return std::make_tuple(newW, newH);
    };

    for (int i = 0; i < 4; i++) {
        int xOff, yOff;
        float scale;
        auto [newW, newH] = box(i, xOff, yOff, scale);

        cv::Mat resized;
        cv::resize(samples[i].image, resized, cv::Size(newW, newH));

        // 计算有效粘贴区域
        int x1 = std::max(0, xOff), y1 = std::max(0, yOff);
        int x2 = std::min(outSize, xOff + newW), y2 = std::min(outSize, yOff + newH);
        int sx1 = x1 - xOff, sy1 = y1 - yOff;
        int sx2 = sx1 + (x2 - x1), sy2 = sy1 + (y2 - y1);

        if (sx2 <= sx1 || sy2 <= sy1) continue;
        resized(cv::Rect(sx1, sy1, sx2 - sx1, sy2 - sy1))
            .copyTo(result.image(cv::Rect(x1, y1, x2 - x1, y2 - y1)));

        // 调整并裁剪 box 坐标
        for (size_t j = 0; j < samples[i].boxes.size(); j++) {
            auto& b = samples[i].boxes[j];
            float absCx = (b.x * samples[i].image.cols) * scale + xOff;
            float absCy = (b.y * samples[i].image.rows) * scale + yOff;
            float absW  = b.width * samples[i].image.cols * scale;
            float absH  = b.height * samples[i].image.rows * scale;

            // 裁剪到画布边界
            float boxX1 = std::max(0.0f, absCx - absW / 2);
            float boxY1 = std::max(0.0f, absCy - absH / 2);
            float boxX2 = std::min((float)outSize, absCx + absW / 2);
            float boxY2 = std::min((float)outSize, absCy + absH / 2);
            float clippedW = boxX2 - boxX1;
            float clippedH = boxY2 - boxY1;
            if (clippedW <= 0 || clippedH <= 0) continue;

            result.boxes.push_back(cv::Rect2f(
                (boxX1 + boxX2) / 2 / outSize,  // 重新归一化 cx
                (boxY1 + boxY2) / 2 / outSize,  // cy
                clippedW / outSize,              // w
                clippedH / outSize));            // h
            result.labels.push_back(samples[i].labels[j]);
        }
    }
    return result;
}

// ============================================================
// HSV Jitter
// ============================================================
YOLOSample hsvJitter(const YOLOSample& sample, float hGain, float sGain, float vGain) {
    if (sample.image.empty()) return sample;

    std::mt19937 rng(std::random_device{}());
    float hVar = std::uniform_real_distribution(-hGain, hGain)(rng);
    float sVar = std::uniform_real_distribution(-sGain, sGain)(rng);
    float vVar = std::uniform_real_distribution(-vGain, vGain)(rng);

    cv::Mat hsv;
    cv::cvtColor(sample.image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    channels[0] += hVar;  // H
    // S 和 V 用缩放而非加法 (更稳定)
    float sScale = 1 + sVar / 100;
    float vScale = 1 + vVar / 100;
    channels[1] = cv::min(cv::max(channels[1] * sScale, 0.0), 255.0);
    channels[2] = cv::min(cv::max(channels[2] * vScale, 0.0), 255.0);

    cv::merge(channels, hsv);
    YOLOSample result = sample;
    cv::cvtColor(hsv, result.image, cv::COLOR_HSV2BGR);
    return result;
}

// ============================================================
// Random Flip H
// ============================================================
YOLOSample randomFlipH(const YOLOSample& sample, float prob) {
    std::mt19937 rng(std::random_device{}());
    if (std::uniform_real_distribution<float>(0, 1)(rng) >= prob) return sample;

    YOLOSample result = sample;
    cv::flip(sample.image, result.image, 1); // horizontal
    result.boxes.clear();
    for (auto& b : sample.boxes) {
        // cx → 1 - cx
        result.boxes.push_back(cv::Rect2f(1 - b.x, b.y, b.width, b.height));
    }
    result.labels = sample.labels;
    return result;
}

// ============================================================
// Letterbox: 等比例缩放 + pad 到 targetSize×targetSize
// ============================================================
YOLOSample letterbox(const YOLOSample& sample, int targetSize) {
    if (sample.image.empty()) return sample;

    int h = sample.image.rows, w = sample.image.cols;
    float scale = std::min((float)targetSize / w, (float)targetSize / h);
    int newW = int(w * scale), newH = int(h * scale);
    int padX = (targetSize - newW) / 2, padY = (targetSize - newH) / 2;

    cv::Mat resized;
    cv::resize(sample.image, resized, cv::Size(newW, newH));

    cv::Mat out(targetSize, targetSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));

    YOLOSample result;
    result.image = out;
    result.imagePath = sample.imagePath;
    result.labels = sample.labels;
    for (auto& b : sample.boxes) {
        // 调整归一化坐标: 原 box 在 letterbox 后的映射
        float cx = (b.x * w * scale + padX) / targetSize;
        float cy = (b.y * h * scale + padY) / targetSize;
        float bw = b.width * w * scale / targetSize;
        float bh = b.height * h * scale / targetSize;
        result.boxes.push_back(cv::Rect2f(cx, cy, bw, bh));
    }
    return result;
}

// ============================================================
// YOLODataLoader
// ============================================================
YOLODataLoader::YOLODataLoader(std::shared_ptr<YOLODataset> dataset,
    int batchSize, int imgSize, bool shuffle)
    : dataset_(dataset), batchSize_(batchSize), imgSize_(imgSize),
      shuffle_(shuffle), rng_(std::random_device{}()) {
    Reset();
}

void YOLODataLoader::Reset() {
    current_ = 0;
    indices_.resize(dataset_->Size());
    for (size_t i = 0; i < indices_.size(); i++) indices_[i] = i;
    if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
}

bool YOLODataLoader::HasNext() const {
    return current_ < indices_.size();
}

size_t YOLODataLoader::NumBatches() const {
    return (indices_.size() + batchSize_ - 1) / batchSize_;
}

std::vector<YOLOSample> YOLODataLoader::loadBatch() {
    std::vector<YOLOSample> batch;
    int count = 0;
    while (count < batchSize_ && current_ < indices_.size()) {
        auto sample = dataset_->Get(indices_[current_++]);

        // 应用增强
        if (mosaic && batch.size() == 0 && current_ >= 4) {
            // mosaic: 取前 4 个样本 (包括当前) 合成 1 张
            std::vector<YOLOSample> mosaicSamples;
            for (int k = 0; k < 4 && current_ - 1 - k < indices_.size(); k++) {
                mosaicSamples.push_back(dataset_->Get(indices_[current_ - 1 - k]));
            }
            // mosaicSamples 可能与 batch 中已有的重复，简化: 只对第一个样本做 mosaic
            sample = mosaicAugment(
                { dataset_->Get(indices_[current_ - 1]),
                  dataset_->Get(indices_[(current_ - 1 + 1) % indices_.size()]),
                  dataset_->Get(indices_[(current_ - 1 + 2) % indices_.size()]),
                  dataset_->Get(indices_[(current_ - 1 + 3) % indices_.size()]) },
                imgSize_);
        } else {
            sample = letterbox(sample, imgSize_);
        }

        if (flip) sample = randomFlipH(sample);
        if (hsvJitterOn) sample = aicore::hsvJitter(sample);

        batch.push_back(sample);
        count++;
    }
    return batch;
}

YOLOBatch YOLODataLoader::collate(const std::vector<YOLOSample>& samples) {
    YOLOBatch batch;
    batch.numClasses = dataset_->NumClasses();

    int B = (int)samples.size();
    int H = imgSize_, W = imgSize_;
    batch.images = torch::empty({B, 3, H, W}, torch::kByte);
    std::vector<torch::Tensor> targetList;

    for (int i = 0; i < B; i++) {
        auto& s = samples[i];
        // BGR → RGB uint8 tensor [3, H, W]
        cv::Mat rgb;
        cv::cvtColor(s.image, rgb, cv::COLOR_BGR2RGB);
        auto imgTensor = torch::from_blob(rgb.data, {H, W, 3}, torch::kByte)
            .permute({2, 0, 1}).clone(); // HWC→CHW
        batch.images[i] = imgTensor;

        for (size_t j = 0; j < s.boxes.size(); j++) {
            // cx,cy → xyxy → 实际 box 格式: YOLOLoss 需要 (bi, cls, cx, cy, w, h)
            // boxes 已经是 (cx,cy,w,h) 归一化
            auto t = torch::tensor({
                (float)i,                    // batchIdx
                (float)s.labels[j],          // cls
                s.boxes[j].x,                // cx 归一化
                s.boxes[j].y,                // cy 归一化
                s.boxes[j].width,            // w 归一化
                s.boxes[j].height            // h 归一化
            });
            targetList.push_back(t);
        }
    }

    if (targetList.empty()) {
        batch.targets = torch::empty({0, 6});
    } else {
        batch.targets = torch::stack(targetList);
    }
    return batch;
}

YOLOBatch YOLODataLoader::Next() {
    auto samples = loadBatch();
    return collate(samples);
}

} // namespace aicore
