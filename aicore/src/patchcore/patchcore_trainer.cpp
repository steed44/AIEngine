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
    if (cfg.backboneType == "model_backend") {
        return Status{StatusCode::ErrorInternal,
            "patchcore: model_backend training not supported (stub), use opencv_dnn"};
    }

    if (dataset.Size() == 0) {
        return Status{StatusCode::ErrorInvalidInput, "patchcore: dataset is empty"};
    }

    auto layerNames = SplitLayerNames(cfg.backboneLayers);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    std::vector<PatchFeature> allFeatures;

    for (size_t i = 0; i < dataset.Size(); i++) {
        auto sample = dataset.Get(i);
        auto feats = ExtractPatchFeatures(net, sample.image, cfg.inputSize, layerNames);
        allFeatures.insert(allFeatures.end(), feats.begin(), feats.end());
    }

    if (allFeatures.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: no features extracted"};
    }

    if (allFeatures.size() > cfg.maxFeatures) {
        std::shuffle(allFeatures.begin(), allFeatures.end(),
                     std::mt19937(std::random_device()()));
        allFeatures.resize(cfg.maxFeatures);
    }

    size_t targetSize = static_cast<size_t>(allFeatures.size() * cfg.coresetFraction);
    if (targetSize == 0) targetSize = 1;

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
