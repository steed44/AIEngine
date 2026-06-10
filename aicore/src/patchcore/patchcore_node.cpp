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

    cv::Mat img = inputs[0].image;
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

    cv::Mat scoreMap(img.rows, img.cols, CV_32F, anomalyMap.data());

    double maxVal = 0;
    cv::minMaxLoc(scoreMap, nullptr, &maxVal);

    Frame out(scoreMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
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
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputSize_, inputSize_));
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0 / 255);

    Tensor input;
    input.dtype = DataType::kFloat32;
    input.shape = {1, 3, inputSize_, inputSize_};
    input.bytes = 1 * 3 * inputSize_ * inputSize_ * sizeof(float);
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
