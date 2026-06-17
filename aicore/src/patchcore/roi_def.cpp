// ============================================================
// roi_def.cpp — 多 ROI 配置 JSON 序列化实现
// 功能：解析/生成 MultiRoiConfig 的 JSON 格式（nlohmann-json）
// ============================================================
#include "patchcore/roi_def.h"
#include <fstream>
#include <sstream>

// 使用 json.hpp（nlohmann-json 单头文件）
#include "json.hpp"
using json = nlohmann::json;

namespace aicore {

// -------------------------------------------------------
// 从 JSON 文件加载 MultiRoiConfig
// 格式示例：
// {
//   "backbone": {
//     "type": "libtorch",
//     "model_path": "patchcore_backbone.pt",
//     "input_size": 224,
//     "layers": "layer2,layer3"
//   },
//   "rois": [
//     {"id": "connector", "x": 10, "y": 20, "width": 120, "height": 80},
//     {"id": "chip", "x": 200, "y": 150, "width": 100, "height": 100}
//   ],
//   "train": { "coreset_fraction": 0.1, "max_features": 100000 },
//   "inference": { "anomaly_threshold": 0.5 },
//   "model_dir": "./models"
// }
// -------------------------------------------------------
Status MultiRoiConfig::FromJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status{StatusCode::ErrorConfigParse,
            "cannot open config file: " + path};
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorConfigParse,
            "JSON parse error: " + std::string(e.what())};
    }

    try {
        // ---- 解析 backbone 配置 ----
        auto& bk = j["backbone"];
        backboneType = bk.value("type", backboneType);
        backendType = bk.value("backend_type", backendType);
        backboneModelPath = bk.value("model_path", "");
        backboneLayers = bk.value("layers", backboneLayers);
        inputSize = bk.value("input_size", inputSize);

        // ---- 解析 ROI 列表 ----
        rois.clear();
        for (auto& r : j["rois"]) {
            RoiDef roi;
            roi.id  = r.value("id", "");
            roi.x   = r.value("x", 0);
            roi.y   = r.value("y", 0);
            roi.w   = r.value("width", 0);
            roi.h   = r.value("height", 0);
            if (roi.id.empty() || roi.w <= 0 || roi.h <= 0) {
                return Status{StatusCode::ErrorConfigParse,
                    "invalid ROI definition: id and w/h required"};
            }
            rois.push_back(roi);
        }

        if (rois.empty()) {
            return Status{StatusCode::ErrorConfigParse,
                "at least one ROI must be defined"};
        }

        // ---- 解析训练参数 ----
        if (j.contains("train")) {
            auto& tr = j["train"];
            coresetFraction = tr.value("coreset_fraction", coresetFraction);
            maxFeatures = tr.value("max_features", maxFeatures);
        }

        // ---- 解析推理参数 ----
        if (j.contains("inference")) {
            auto& inf = j["inference"];
            anomalyThreshold = inf.value("anomaly_threshold", anomalyThreshold);
        }

        // ---- 模型目录 ----
        modelDir = j.value("model_dir", modelDir);
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorConfigParse,
            "config field error: " + std::string(e.what())};
    }

    return Status{};
}

// -------------------------------------------------------
// 序列化为 JSON 字符串（用于调试和验证）
// -------------------------------------------------------
std::string MultiRoiConfig::ToJson() const {
    json j;
    j["backbone"]["type"] = backboneType;
    j["backbone"]["backend_type"] = backendType;
    j["backbone"]["model_path"] = backboneModelPath;
    j["backbone"]["layers"] = backboneLayers;
    j["backbone"]["input_size"] = inputSize;

    for (auto& r : rois) {
        json rj;
        rj["id"] = r.id;
        rj["x"] = r.x;
        rj["y"] = r.y;
        rj["width"] = r.w;
        rj["height"] = r.h;
        j["rois"].push_back(rj);
    }

    j["train"]["coreset_fraction"] = coresetFraction;
    j["train"]["max_features"] = maxFeatures;
    j["inference"]["anomaly_threshold"] = anomalyThreshold;
    j["model_dir"] = modelDir;

    return j.dump(2);
}

} // namespace aicore
