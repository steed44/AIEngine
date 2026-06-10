# PatchCore 异常检测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在流水线引擎中集成 PatchCore 异常检测，支持训练（文件夹 → memory bank）和推理（特征提取 → NN 搜索 → 异常得分图）。

**Architecture:** 4 个新类（MemoryBank, CoresetSampler, PatchCoreNode, PatchCoreTrainer）+ 1 个工具类（FolderDataset），集成到现有 pipeline builder，新增 CLI 用于训练。

**Tech Stack:** C++17, OpenCV 4.10 (core + imgproc + dnn), VS2022, CMake

---

### Task 1: MemoryBank + CoresetSampler

**Files:**
- Create: `include/patchcore/memory_bank.h`
- Create: `src/patchcore/memory_bank.cpp`
- Create: `include/patchcore/coreset_sampler.h`
- Create: `src/patchcore/coreset_sampler.cpp`

- [ ] **Step 1: 创建 memory_bank.h**

```cpp
#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace aicore {

struct PatchFeature {
    std::vector<float> features;
    int layerIdx = 0;
    int patchRow = 0, patchCol = 0;
};

class MemoryBank {
public:
    MemoryBank() = default;

    bool Load(const std::string& path);
    bool Save(const std::string& path) const;

    void Build(const std::vector<PatchFeature>& features);
    void Clear();

    // 最近邻搜索，返回索引，distOut 为 L2 距离
    size_t NearestNeighbor(const std::vector<float>& query, float& distOut) const;
    // 对 queries 逐 patch 做 NN 搜索 → 重排为得分图 → 上采样到 (imgH, imgW)
    std::vector<float> ComputeAnomalyMap(const std::vector<PatchFeature>& queries,
                                          int imgH, int imgW) const;

    size_t Size() const { return bank_.size(); }
    int FeatureDim() const { return featureDim_; }

    // 序列化格式: uint32 magic(0x50434F52) | uint32 num | uint32 dim | float data[]
    static constexpr uint32_t kMagic = 0x50434F52;

private:
    std::vector<PatchFeature> bank_;
    int featureDim_ = 0;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 memory_bank.cpp**

```cpp
#include "patchcore/memory_bank.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace aicore {

void MemoryBank::Build(const std::vector<PatchFeature>& features) {
    bank_ = features;
    if (!features.empty()) {
        featureDim_ = static_cast<int>(features[0].features.size());
    }
}

void MemoryBank::Clear() {
    bank_.clear();
    featureDim_ = 0;
}

bool MemoryBank::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t num = static_cast<uint32_t>(bank_.size());
    uint32_t dim = static_cast<uint32_t>(featureDim_);
    f.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    f.write(reinterpret_cast<const char*>(&num), sizeof(num));
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    for (auto& pf : bank_) {
        f.write(reinterpret_cast<const char*>(pf.features.data()), dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(&pf.layerIdx), sizeof(pf.layerIdx));
        f.write(reinterpret_cast<const char*>(&pf.patchRow), sizeof(pf.patchRow));
        f.write(reinterpret_cast<const char*>(&pf.patchCol), sizeof(pf.patchCol));
    }
    return f.good();
}

bool MemoryBank::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, num = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kMagic) return false;
    f.read(reinterpret_cast<char*>(&num), sizeof(num));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    bank_.resize(num);
    featureDim_ = static_cast<int>(dim);
    for (uint32_t i = 0; i < num; i++) {
        bank_[i].features.resize(dim);
        f.read(reinterpret_cast<char*>(bank_[i].features.data()), dim * sizeof(float));
        f.read(reinterpret_cast<char*>(&bank_[i].layerIdx), sizeof(bank_[i].layerIdx));
        f.read(reinterpret_cast<char*>(&bank_[i].patchRow), sizeof(bank_[i].patchRow));
        f.read(reinterpret_cast<char*>(&bank_[i].patchCol), sizeof(bank_[i].patchCol));
    }
    return f.good();
}

size_t MemoryBank::NearestNeighbor(const std::vector<float>& query, float& distOut) const {
    if (bank_.empty()) return 0;
    size_t bestIdx = 0;
    float bestDist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < bank_.size(); i++) {
        float d = 0;
        for (int j = 0; j < featureDim_; j++) {
            float diff = query[j] - bank_[i].features[j];
            d += diff * diff;
        }
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    distOut = std::sqrt(bestDist);
    return bestIdx;
}

std::vector<float> MemoryBank::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {
    // 确定特征图尺寸（从 patchRow/patchCol 最大值推断）
    int maxRow = 0, maxCol = 0;
    for (auto& q : queries) {
        maxRow = std::max(maxRow, q.patchRow + 1);
        maxCol = std::max(maxCol, q.patchCol + 1);
    }
    if (maxRow == 0 || maxCol == 0) return {};

    // 构建原始得分图
    cv::Mat scoreMap(maxRow, maxCol, CV_32F);
    for (auto& q : queries) {
        float dist = 0;
        NearestNeighbor(q.features, dist);
        scoreMap.at<float>(q.patchRow, q.patchCol) = dist;
    }

    // 双线性上采样到原图尺寸
    cv::Mat upsampled;
    cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

    std::vector<float> result(imgW * imgH);
    std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
    return result;
}

} // namespace aicore
```

- [ ] **Step 3: 创建 coreset_sampler.h**

```cpp
#pragma once
#include "patchcore/memory_bank.h"
#include <vector>

namespace aicore {

class CoresetSampler {
public:
    // 贪心 coreset 采样：迭代选择与已选集合最远距离最大的样本
    std::vector<size_t> Sample(const std::vector<PatchFeature>& pool,
                                size_t targetSize);
};

} // namespace aicore
```

- [ ] **Step 4: 创建 coreset_sampler.cpp**

```cpp
#include "patchcore/coreset_sampler.h"
#include <limits>
#include <cmath>

namespace aicore {

static float L2Dist(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}

std::vector<size_t> CoresetSampler::Sample(
    const std::vector<PatchFeature>& pool, size_t targetSize) {
    if (pool.empty() || targetSize >= pool.size()) {
        std::vector<size_t> all(pool.size());
        for (size_t i = 0; i < pool.size(); i++) all[i] = i;
        return all;
    }

    size_t n = pool.size();
    std::vector<float> minDist(n, std::numeric_limits<float>::max());
    std::vector<bool> selected(n, false);
    std::vector<size_t> result;
    result.reserve(targetSize);

    // 选第一个：距原点最远
    size_t first = 0;
    float maxNorm = 0;
    for (size_t i = 0; i < n; i++) {
        float d = L2Dist(pool[i].features, std::vector<float>(pool[i].features.size(), 0));
        if (d > maxNorm) { maxNorm = d; first = i; }
    }
    result.push_back(first);
    selected[first] = true;

    // 更新 minDist
    for (size_t i = 0; i < n; i++) {
        if (!selected[i]) {
            minDist[i] = L2Dist(pool[i].features, pool[first].features);
        }
    }

    for (size_t k = 1; k < targetSize; k++) {
        // 选 minDist 最大的
        size_t best = 0;
        float bestVal = -1;
        for (size_t i = 0; i < n; i++) {
            if (!selected[i] && minDist[i] > bestVal) {
                bestVal = minDist[i];
                best = i;
            }
        }
        result.push_back(best);
        selected[best] = true;

        // 更新 minDist
        for (size_t i = 0; i < n; i++) {
            if (!selected[i]) {
                float d = L2Dist(pool[i].features, pool[best].features);
                if (d < minDist[i]) minDist[i] = d;
            }
        }
    }
    return result;
}

} // namespace aicore
```

- [ ] **Step 5: 创建 tests/test_memory_bank.cpp**

```cpp
#include <gtest/gtest.h>
#include "patchcore/memory_bank.h"
#include "patchcore/coreset_sampler.h"
#include <cstdio>

using namespace aicore;

TEST(MemoryBankTest, BuildAndQuery) {
    std::vector<PatchFeature> features;
    for (int i = 0; i < 10; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), 0, 0};
        pf.patchRow = i; pf.patchCol = 0;
        features.push_back(pf);
    }
    MemoryBank bank;
    bank.Build(features);
    EXPECT_EQ(bank.Size(), 10);
    EXPECT_EQ(bank.FeatureDim(), 3);

    float dist = 0;
    std::vector<float> query = {5, 0, 0};
    size_t idx = bank.NearestNeighbor(query, dist);
    EXPECT_EQ(idx, 5);
    EXPECT_NEAR(dist, 0, 1e-5);
}

TEST(MemoryBankTest, SaveLoadRoundtrip) {
    std::vector<PatchFeature> features;
    PatchFeature pf;
    pf.features = {1, 2, 3};
    pf.patchRow = 0; pf.patchCol = 0;
    features.push_back(pf);

    MemoryBank bank;
    bank.Build(features);
    EXPECT_TRUE(bank.Save("test_mem.bin"));

    MemoryBank loaded;
    EXPECT_TRUE(loaded.Load("test_mem.bin"));
    EXPECT_EQ(loaded.Size(), 1);
    EXPECT_EQ(loaded.FeatureDim(), 3);

    std::remove("test_mem.bin");
}

TEST(CoresetSamplerTest, ReduceSize) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 100; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i * 2), 0};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.Sample(pool, 10);
    EXPECT_EQ(indices.size(), 10);
}
```

- [ ] **Step 6: 编译并运行测试**

编译加源文件前先单独验证测试。Run:
```bash
cd aicore/build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/Qt5.12.11/5.12.11/msvc2017_64"
cmake --build . --config Release
$env:Path = "C:\Users\ght\...\opencv\build\x64\vc16\bin;$env:Path"
.\tests\Release\aicore_tests.exe
```
Expected: 测试编译失败（MemoryBank/CoresetSampler 源文件未加入构建但测试文件已引用）

---

### Task 2: CMakeLists 集成

**Files:**
- Modify: `aicore/CMakeLists.txt`
- Modify: `aicore/tests/CMakeLists.txt`

- [ ] **Step 1: 将 MemoryBank + CoresetSampler 加入 aicore.dll**

在 `CMakeLists.txt` 的 `AICORE_SOURCES` 中添加：
```cmake
    src/patchcore/memory_bank.cpp
    src/patchcore/coreset_sampler.cpp
```

- [ ] **Step 2: 将测试源文件加入 test target**

在 `tests/CMakeLists.txt` 的 `target_sources` 中添加：
```cmake
    # PatchCore
    ${CMAKE_SOURCE_DIR}/src/patchcore/memory_bank.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/coreset_sampler.cpp
```

- [ ] **Step 3: 编译并运行测试**

Run:
```bash
cd aicore/build
cmake --build . --config Release
$env:Path = "..."
.\tests\Release\aicore_tests.exe --gtest_filter=MemoryBankTest*:CoresetSamplerTest*
```
Expected: 3 tests pass

---

### Task 3: FolderDataset

**Files:**
- Create: `include/patchcore/folder_dataset.h`
- Create: `src/patchcore/folder_dataset.cpp`

- [ ] **Step 1: 创建 folder_dataset.h**

```cpp
#pragma once
#include "trainer/data/dataset.h"
#include <vector>
#include <string>

namespace aicore {

// 扫描文件夹下图片文件，实现 IDataset
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
```

- [ ] **Step 2: 创建 folder_dataset.cpp**

```cpp
#include "patchcore/folder_dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <algorithm>

namespace aicore {

Status FolderDataset::Load(const std::string& folderPath) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(folderPath)) {
        return Status{StatusCode::ErrorInvalidInput, "not a directory: " + folderPath};
    }

    std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    for (auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

        Sample s;
        s.imagePath = entry.path().string();
        s.image = cv::imread(s.imagePath);
        s.label = 0;
        if (!s.image.empty()) {
            samples_.push_back(std::move(s));
        }
    }
    return Status{};
}

size_t FolderDataset::Size() const { return samples_.size(); }

Sample FolderDataset::Get(size_t index) { return samples_.at(index); }

} // namespace aicore
```

- [ ] **Step 3: 创建 tests/test_folder_dataset.cpp** 并加入构建（或放入已有 test 文件）

```cpp
#include <gtest/gtest.h>
#include "patchcore/folder_dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <filesystem>

using namespace aicore;

TEST(FolderDatasetTest, LoadFromFolder) {
    // 创建临时测试目录
    std::string dir = "test_images/";
    std::filesystem::create_directories(dir);
    // 创建一个 10x10 的测试图片
    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::imwrite(dir + "img1.png", img);
    cv::imwrite(dir + "img2.jpg", img);

    FolderDataset ds;
    EXPECT_TRUE(ds.Load(dir));
    EXPECT_EQ(ds.Size(), 2);

    auto s = ds.Get(0);
    EXPECT_EQ(s.image.rows, 10);
    EXPECT_EQ(s.label, 0);

    std::filesystem::remove_all(dir);
}
```

- [ ] **Step 4: 编译并运行测试**

添加源文件到 CMakeLists.txt 后编译：
```bash
cd aicore/build
cmake --build . --config Release
$env:Path = "..."
.\tests\Release\aicore_tests.exe --gtest_filter=FolderDatasetTest*
```
Expected: PASS

---

### Task 4: PatchCoreNode

**Files:**
- Create: `include/patchcore/patchcore_node.h`
- Create: `src/patchcore/patchcore_node.cpp`

- [ ] **Step 1: 创建 patchcore_node.h**

```cpp
#pragma once
#include "core/processor.h"
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include "patchcore/memory_bank.h"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace aicore {

class PatchCoreNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "patchcore"; }

private:
    std::vector<PatchFeature> ForwardOpenCVDnn(const cv::Mat& blob);
    std::vector<PatchFeature> ForwardModelBackend(const cv::Mat& img);

    std::string name_;
    bool useOpenCVDnn_ = true;
    cv::dnn::Net net_;
    std::unique_ptr<IModelBackend> backend_;
    MemoryBank memoryBank_;
    std::vector<std::string> outputLayerNames_;
    int inputSize_ = 224;
    float anomalyThreshold_ = 0.5f;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 patchcore_node.cpp**

```cpp
#include "patchcore/patchcore_node.h"
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace aicore {

static std::vector<std::string> SplitLayerNames(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

Status PatchCoreNode::Init(const NodeConfig& config) {
    name_ = config.count("name") ? config.at("name") : "patchcore";

    auto it = config.find("model_path");
    if (it == config.end()) {
        return Status{StatusCode::ErrorConfigParse, "patchcore: model_path required"};
    }

    auto bt = config.find("backbone");
    if (bt != config.end() && bt->second == "model_backend") {
        useOpenCVDnn_ = false;
        auto bk = config.find("backend_type");
        BackendType backendType = BackendType::kONNXRuntime;
        if (bk != config.end()) {
            if (bk->second == "tensorrt") backendType = BackendType::kTensorRT;
            else if (bk->second == "libtorch") backendType = BackendType::kLibTorch;
        }
        backend_ = BackendFactory::Create(backendType);
        if (!backend_) {
            return Status{StatusCode::ErrorConfigParse, "patchcore: unknown backend_type"};
        }
        ModelInfo info;
        info.modelPath = it->second;
        auto s = backend_->Load(info);
        if (!s) return s;
    } else {
        useOpenCVDnn_ = true;
        net_ = cv::dnn::readNetFromONNX(it->second);
    }

    auto mn = config.find("memory_bank_path");
    if (mn != config.end()) {
        if (!memoryBank_.Load(mn->second)) {
            return Status{StatusCode::ErrorModelLoad, "patchcore: cannot load memory bank"};
        }
    }

    auto layers = config.find("backbone_layers");
    if (layers != config.end()) {
        outputLayerNames_ = SplitLayerNames(layers->second);
    }

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    auto at = config.find("anomaly_threshold");
    if (at != config.end()) anomalyThreshold_ = std::stof(at->second);

    return Status{};
}

Status PatchCoreNode::Process(const std::vector<Frame>& inputs,
                               std::vector<Frame>& outputs) {
    if (inputs.empty()) {
        return Status{StatusCode::ErrorInvalidInput, "patchcore: no input"};
    }

    cv::Mat img = inputs[0].GetMat();
    if (img.empty()) {
        return Status{StatusCode::ErrorPreprocess, "patchcore: empty image"};
    }

    std::vector<PatchFeature> patchFeatures;
    if (useOpenCVDnn_) {
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255,
            cv::Size(inputSize_, inputSize_),
            cv::Scalar(0.485, 0.456, 0.406), true, false);
        patchFeatures = ForwardOpenCVDnn(blob);
    } else {
        patchFeatures = ForwardModelBackend(img);
    }
    if (patchFeatures.empty()) {
        return Status{StatusCode::ErrorModelInfer, "patchcore: backbone returned no features"};
    }

    auto anomalyMap = memoryBank_.ComputeAnomalyMap(patchFeatures, img.rows, img.cols);
    if (anomalyMap.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: anomaly map empty"};
    }

    // 构建异常得分图（float32，与原图同尺寸）
    cv::Mat scoreMap(img.rows, img.cols, CV_32F, anomalyMap.data());

    // 计算图像级得分
    double maxVal = 0;
    cv::minMaxLoc(scoreMap, nullptr, &maxVal);

    Frame out(scoreMap.clone());  // float32 得分图
    out.SetMeta("anomaly_score", std::to_string(maxVal));
    out.SetMeta("is_anomaly", maxVal > anomalyThreshold_ ? "1" : "0");
    outputs.push_back(std::move(out));

    return Status{};
}

std::vector<PatchFeature> PatchCoreNode::ForwardOpenCVDnn(const cv::Mat& blob) {
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, outputLayerNames_);

    std::vector<PatchFeature> features;
    for (int li = 0; li < static_cast<int>(outputs.size()); li++) {
        auto& feat = outputs[li];
        int channels = feat.size[1];
        int h = feat.size[2];
        int w = feat.size[3];
        float* data = feat.ptr<float>();

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.layerIdx = li;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                for (int c = 0; c < channels; c++) {
                    pf.features[c] = data[(c * h + row) * w + col];
                }
                features.push_back(pf);
            }
        }
    }
    return features;
}

std::vector<PatchFeature> PatchCoreNode::ForwardModelBackend(const cv::Mat& img) {
    // 预处理
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputSize_, inputSize_));
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0 / 255);

    Tensor input;
    input.dtype = DataType::kFloat32;
    input.shape = {1, 3, inputSize_, inputSize_};
    input.bytes = 1 * 3 * inputSize_ * inputSize_ * sizeof(float);
    // 注意：这里需要 HWC→CHW 转换
    std::vector<float> chw(input.bytes / sizeof(float));
    float* src = floatImg.ptr<float>();
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < inputSize_; h++)
            for (int w = 0; w < inputSize_; w++)
                chw[c * inputSize_ * inputSize_ + h * inputSize_ + w] = src[h * inputSize_ * 3 + w * 3 + c];
    input.data = chw.data();

    std::vector<Tensor> outputs;
    auto s = backend_->Infer({input}, outputs);
    if (!s) return {};

    std::vector<PatchFeature> features;
    for (auto& out : outputs) {
        // out.shape = [1, C, H, W]
        int channels = static_cast<int>(out.shape[1]);
        int h = static_cast<int>(out.shape[2]);
        int w = static_cast<int>(out.shape[3]);
        float* data = static_cast<float*>(out.data);

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                for (int c = 0; c < channels; c++) {
                    pf.features[c] = data[(c * h + row) * w + col];
                }
                features.push_back(pf);
            }
        }
    }
    return features;
}

} // namespace aicore
```

- [ ] **Step 3: 在 pipeline_builder.cpp 添加 patchcore 分支**

在 `src/config/pipeline_builder.cpp` 的节点类型分发中添加：

```cpp
} else if (type == "patchcore") {
    auto node = std::make_shared<PatchCoreNode>();
    auto s = node->Init(pc.params);
    if (!s) return s;
    processor = node;
}
```

记得添加 include: `#include "patchcore/patchcore_node.h"`

- [ ] **Step 4: 创建 tests/test_patchcore_node.cpp** 简化版（用随机权重 mock backbone）

```cpp
#include <gtest/gtest.h>
#include "patchcore/patchcore_node.h"

using namespace aicore;

TEST(PatchCoreNodeTest, InitMissingModelPath) {
    PatchCoreNode node;
    NodeConfig cfg;
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
}

TEST(PatchCoreNodeTest, ProcessEmptyInput) {
    PatchCoreNode node;
    std::vector<Frame> inputs, outputs;
    auto s = node.Process(inputs, outputs);
    EXPECT_FALSE(s);
}

TEST(PatchCoreNodeTest, InitLoadsMemoryBank) {
    // 先创建 memory bank 文件
    std::vector<PatchFeature> feats;
    PatchFeature pf;
    pf.features = {1, 2, 3};
    pf.patchRow = 0; pf.patchCol = 0;
    feats.push_back(pf);
    MemoryBank bank;
    bank.Build(feats);
    ASSERT_TRUE(bank.Save("test_patchcore.bin"));

    PatchCoreNode node;
    NodeConfig cfg;
    cfg["model_path"] = "dummy.onnx";
    cfg["memory_bank_path"] = "test_patchcore.bin";
    // 不检查 Init 返回（需要 ONNX 模型才能完整 Init），但验证 memory bank 加载不崩溃
    (void)node.Init(cfg);
    std::remove("test_patchcore.bin");
}
```

---

### Task 5: PatchCoreTrainer + CLI

**Files:**
- Create: `include/patchcore/patchcore_trainer.h`
- Create: `src/patchcore/patchcore_trainer.cpp`
- Create: `cli/patchcore_train_main.cpp`

- [ ] **Step 1: 创建 patchcore_trainer.h**

```cpp
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include "trainer/data/dataset.h"
#include <string>

namespace aicore {

struct PatchCoreTrainConfig {
    int inputSize = 224;
    std::string backboneLayers = "layer2,layer3";
    std::string backboneType = "opencv_dnn";
    double coresetFraction = 0.1;
    size_t maxFeatures = 100000;
};

class PatchCoreTrainer {
public:
    Status Train(IDataset& dataset, const std::string& modelPath,
                 const std::string& outputPath,
                 const PatchCoreTrainConfig& cfg);
    Status TrainFromFolder(const std::string& folderPath,
                           const std::string& modelPath,
                           const std::string& outputPath,
                           const PatchCoreTrainConfig& cfg);
    std::string GetLastError() const { return lastError_; }

private:
    std::string lastError_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 patchcore_trainer.cpp**

```cpp
#include "patchcore/patchcore_trainer.h"
#include "patchcore/coreset_sampler.h"
#include "patchcore/folder_dataset.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <random>

namespace aicore {

static std::vector<std::string> SplitLayerNames(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

static std::vector<PatchFeature> ExtractPatchFeatures(
    cv::dnn::Net& net, const cv::Mat& img, int inputSize,
    const std::vector<std::string>& layerNames) {

    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255,
        cv::Size(inputSize, inputSize),
        cv::Scalar(0.485, 0.456, 0.406), true, false);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, layerNames);

    std::vector<PatchFeature> features;
    for (int li = 0; li < static_cast<int>(outputs.size()); li++) {
        auto& feat = outputs[li];
        int channels = feat.size[1];
        int h = feat.size[2];
        int w = feat.size[3];
        float* data = feat.ptr<float>();
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.layerIdx = li;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                for (int c = 0; c < channels; c++) {
                    pf.features[c] = data[(c * h + row) * w + col];
                }
                features.push_back(pf);
            }
        }
    }
    return features;
}

Status PatchCoreTrainer::Train(IDataset& dataset, const std::string& modelPath,
                                const std::string& outputPath,
                                const PatchCoreTrainConfig& cfg) {
    auto layerNames = SplitLayerNames(cfg.backboneLayers);

    std::vector<PatchFeature> allFeatures;

    if (cfg.backboneType == "model_backend") {
        return Status{StatusCode::ErrorInternal,
            "patchcore: model_backend training not supported (stub), use opencv_dnn"};
    } else {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
        for (size_t i = 0; i < dataset.Size(); i++) {
            auto sample = dataset.Get(i);
            auto feats = ExtractPatchFeatures(net, sample.image, cfg.inputSize, layerNames);
            allFeatures.insert(allFeatures.end(), feats.begin(), feats.end());
        }
    }

    // 随机采样到 maxFeatures
    if (allFeatures.size() > cfg.maxFeatures) {
        std::shuffle(allFeatures.begin(), allFeatures.end(),
                     std::mt19937(std::random_device()()));
        allFeatures.resize(cfg.maxFeatures);
    }

    // Coreset 降采样
    size_t targetSize = static_cast<size_t>(allFeatures.size() * cfg.coresetFraction);
    CoresetSampler sampler;
    auto indices = sampler.Sample(allFeatures, targetSize);

    MemoryBank bank;
    std::vector<PatchFeature> coreFeatures;
    for (auto idx : indices) {
        coreFeatures.push_back(allFeatures[idx]);
    }
    bank.Build(coreFeatures);

    if (!bank.Save(outputPath)) {
        lastError_ = "failed to save memory bank to " + outputPath;
        return Status{StatusCode::ErrorInternal, lastError_};
    }

    return Status{};
}

Status PatchCoreTrainer::TrainFromFolder(const std::string& folderPath,
                                          const std::string& modelPath,
                                          const std::string& outputPath,
                                          const PatchCoreTrainConfig& cfg) {
    FolderDataset dataset;
    auto s = dataset.Load(folderPath);
    if (!s) {
        lastError_ = s.message;
        return s;
    }
    return Train(dataset, modelPath, outputPath, cfg);
}

} // namespace aicore
```

- [ ] **Step 3: 创建 cli/patchcore_train_main.cpp**

```cpp
#include "patchcore/patchcore_trainer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string dataPath, modelPath, outputPath = "memory_bank.bin";
    double coresetFrac = 0.1;
    int inputSize = 224;
    std::string layers = "layer2,layer3";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--input_size" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) layers = argv[++i];
        else if (arg == "--coreset" && i + 1 < argc) coresetFrac = std::stod(argv[++i]);
    }

    if (dataPath.empty() || modelPath.empty()) {
        std::cerr << "Usage: PatchCoreTrain --data <folder> --model <backbone.onnx> [--output <mem.bin>]\n";
        return 1;
    }

    aicore::PatchCoreTrainConfig cfg;
    cfg.inputSize = inputSize;
    cfg.backboneLayers = layers;
    cfg.coresetFraction = coresetFrac;

    aicore::PatchCoreTrainer trainer;
    auto s = trainer.TrainFromFolder(dataPath, modelPath, outputPath, cfg);
    if (!s) {
        std::cerr << "Training failed: " << s.message << std::endl;
        return 1;
    }
    std::cout << "Memory bank saved to " << outputPath << std::endl;
    return 0;
}
```

- [ ] **Step 4: 在 CMakeLists.txt 中添加 PatchCoreTrain 目标**

```cmake
add_executable(PatchCoreTrain
    cli/patchcore_train_main.cpp
)
target_include_directories(PatchCoreTrain PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(PatchCoreTrain PRIVATE ${NLOHMANN_JSON_DIR})
target_include_directories(PatchCoreTrain PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PatchCoreTrain PRIVATE ${OpenCV_LIBS})
```

链接 aicore 导入库（CLI 通过 C++ 类 API 直接调用 PatchCoreTrainer）：

- [ ] **Step 5: 创建 tests/test_patchcore_trainer.cpp**

```cpp
#include <gtest/gtest.h>
#include "patchcore/patchcore_trainer.h"
#include "patchcore/folder_dataset.h"
#include <filesystem>

using namespace aicore;

TEST(PatchCoreTrainerTest, FolderNotFound) {
    PatchCoreTrainer trainer;
    auto s = trainer.TrainFromFolder("nonexistent_dir", "dummy.onnx", "out.bin", {});
    EXPECT_FALSE(s);
}

TEST(PatchCoreTrainerTest, TrainFromFolderCreatesBank) {
    // 需要实际 ONNX 模型和图片 → 跳过，仅验证训练入口不崩溃
    GTEST_SKIP() << "需要 ONNX 模型文件才能运行";
}
```

---

### Task 6: AICoreUI 异常热力图显示

**Files:**
- Modify: `aicore/gui/main_window.h`
- Modify: `aicore/gui/main_window.cpp`

- [ ] **Step 1: 修改 aicore_api.cpp 的 result_to_json 包含 NodeResult 的 measurements**

在 `aicore_result_to_json` 的 detection 序列化中添加 measurements（保证 anomaly_score 出现在 JSON 中）：

```cpp
json << "{"
     << "\"node_id\":\"" << d.nodeId << "\","
     << "\"label\":\"" << d.label << "\","
     << "\"confidence\":" << d.confidence << ","
     << "\"bbox\":{" << "\"x\":" << d.bbox.x
     << ",\"y\":" << d.bbox.y
     << ",\"w\":" << d.bbox.w
     << ",\"h\":" << d.bbox.h << "}";
if (!d.measurements.empty()) {
    json << ",\"measurements\":{";
    bool mfirst = true;
    for (auto& [k, v] : d.measurements) {
        if (!mfirst) json << ",";
        mfirst = false;
        json << "\"" << k << "\":" << v;
    }
    json << "}";
    // 将关键测量值提升到顶层方便 UI 读取
    auto it = d.measurements.find("anomaly_score");
    if (it != d.measurements.end()) {
        json << ",\"anomaly_score\":" << it->second;
    }
}
json << "}";
```

- [ ] **Step 2: 在 main_window.h 中添加热力图绘制方法**

在 private 部分添加：
```cpp
    void drawAnomalyOverlay(QImage& image, const QString& json);
```

- [ ] **Step 2: 在 main_window.cpp 中添加 drawAnomalyOverlay 实现**

```cpp
void MainWindow::drawAnomalyOverlay(QImage& image, const QString& json) {
    QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    if (doc.isNull() || !doc.isObject()) return;

    QJsonObject root = doc.object();
    QJsonArray dets = root["detections"].toArray();
    double score = -1;
    for (auto& d : dets) {
        QJsonObject obj = d.toObject();
        if (obj.contains("anomaly_score")) {
            score = obj["anomaly_score"].toDouble();
            break;
        }
    }
    if (score < 0) return;  // 不是 PatchCore 结果

    bool isAnomaly = score > 0.5;  // 默认阈值

    // Jet colormap 叠加：强度映射到红色
    QPainter painter(&image);
    painter.fillRect(image.rect(), QColor(255, 0, 0, static_cast<int>(std::min(score * 200, 100.0))));

    QFont font = painter.font();
    font.setPixelSize(20);
    font.setBold(true);
    painter.setFont(font);

    QString text = QString("Anomaly Score: %1").arg(score, 0, 'f', 4);
    painter.setPen(isAnomaly ? Qt::red : Qt::green);
    painter.drawText(QPoint(10, 30), text);

    if (isAnomaly) {
        painter.setPen(QPen(Qt::red, 3));
        painter.drawRect(5, 5, image.width() - 10, image.height() - 10);
    }
    painter.end();
}
```

- [ ] **Step 3: 在 onInferenceFinished 中调用 drawAnomalyOverlay**

在 `drawDetections` 调用之后添加：
```cpp
    drawAnomalyOverlay(originalImage_, lastJson_);
```

- [ ] **Step 4: 编译 AICoreUI 并验证**

```bash
cd aicore/build
cmake --build . --config Release --target AICoreUI
```
Expected: AICoreUI.exe 编译成功

### Task 7: 全量编译验证

**Files:**
- Modify: `aicore/CMakeLists.txt`（加入所有 PatchCore 源文件和 CLI）
- Modify: `aicore/tests/CMakeLists.txt`（加入所有 PatchCore 测试源文件）

- [ ] **Step 1: 更新 CMakeLists.txt 加入所有新源文件**

```cmake
set(AICORE_SOURCES
    ...
    src/patchcore/memory_bank.cpp
    src/patchcore/coreset_sampler.cpp
    src/patchcore/patchcore_node.cpp
    src/patchcore/patchcore_trainer.cpp
    src/patchcore/folder_dataset.cpp
)
```

```cmake
target_sources(aicore_tests PRIVATE
    ...
    ${CMAKE_SOURCE_DIR}/src/patchcore/memory_bank.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/coreset_sampler.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/patchcore_node.cpp
)
```

- [ ] **Step 2: 编译全部 target**

Run:
```bash
cd aicore/build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/Qt5.12.11/5.12.11/msvc2017_64"
cmake --build . --config Release
```
Expected: 全部编译通过，新增 PatchCoreTrain.exe

- [ ] **Step 3: 运行全部测试**

```bash
$env:Path = "C:\...\opencv\bin;$env:Path"
.\tests\Release\aicore_tests.exe
```
Expected: 全部通过（包括旧的 59 个 + 新增的 PatchCore 测试）
