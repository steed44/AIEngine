// ============================================================
// roi_def.cpp — 多 ROI 配置 JSON 序列化实现
// 功能：解析/生成 MultiRoiConfig 的 JSON 格式（nlohmann-json）
// 支持两种 ROI 模式：
//   1. 固定坐标模式：配置文件定义固定 ROI 坐标
//   2. 每图 ROI 模式：每张图有独立 ROI 坐标 JSON
// ============================================================
#include "patchcore/roi_def.h"
#include <fstream>
#include <sstream>

// 使用 json.hpp（nlohmann-json 单头文件）
#include "json.hpp"
using json = nlohmann::json;

namespace aicore {

// -------------------------------------------------------
// PerImageRoiConfig — JSON 解析
// -------------------------------------------------------
Status PerImageRoiConfig::FromJson(const std::string& path, PerImageRoiConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status{StatusCode::ErrorConfigParse,
            "cannot open per-image ROI config: " + path};
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorConfigParse,
            "JSON parse error in " + path + ": " + std::string(e.what())};
    }

    try {
        out.imagePath = j.value("image", "");
        if (out.imagePath.empty()) {
            // 尝试用文件名作为 key
            out.imagePath = path;
        }

        out.rois.clear();
        if (j.contains("rois")) {
            for (auto& r : j["rois"]) {
                RoiDef roi;
                roi.id  = r.value("id", "");
                roi.x   = r.value("x", 0);
                roi.y   = r.value("y", 0);
                roi.w   = r.value("width", 0);
                roi.h   = r.value("height", 0);
                if (roi.id.empty() || roi.w <= 0 || roi.h <= 0) {
                    return Status{StatusCode::ErrorConfigParse,
                        "invalid ROI in per-image config: id and w/h required"};
                }
                out.rois.push_back(roi);
            }
        }
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorConfigParse,
            "per-image config field error: " + std::string(e.what())};
    }

    return Status{};
}

std::string PerImageRoiConfig::ToJson() const {
    json j;
    j["image"] = imagePath;
    for (auto& r : rois) {
        json rj;
        rj["id"] = r.id;
        rj["x"] = r.x;
        rj["y"] = r.y;
        rj["width"] = r.w;
        rj["height"] = r.h;
        j["rois"].push_back(rj);
    }
    return j.dump(2);
}

// -------------------------------------------------------
// 从 JSON 文件加载 MultiRoiConfig
// 支持两种格式：
//   1. 固定坐标模式（向后兼容）：
//      { "backbone": {...}, "rois": [...], "train": {...}, ... }
//   2. 每图 ROI 模式：
//      { "backbone": {...}, "roi_source": "per_image_file",
//        "per_image_rois_dir": "./roi_annotations/", ... }
//      或
//      { "backbone": {...}, "roi_source": "per_image_list",
//        "per_image_rois_file": "./all_rois.json", ... }
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

        // ---- 解析 ROI 模式 ----
        std::string roiSourceStr = j.value("roi_source", "fixed");
        if (roiSourceStr == "per_image_file") {
            roiSource = RoiSourceType::PerImageFile;
            perImageRoisDir = j.value("per_image_rois_dir", "");
        } else if (roiSourceStr == "per_image_list") {
            roiSource = RoiSourceType::PerImageList;
            perImageRoisFile = j.value("per_image_rois_file", "");
        } else {
            roiSource = RoiSourceType::Fixed;
        }

        // ---- 解析 ROI 列表（固定坐标模式） ----
        rois.clear();
        if (roiSource == RoiSourceType::Fixed && j.contains("rois")) {
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

    switch (roiSource) {
        case RoiSourceType::Fixed:
            j["roi_source"] = "fixed";
            for (auto& r : rois) {
                json rj;
                rj["id"] = r.id;
                rj["x"] = r.x;
                rj["y"] = r.y;
                rj["width"] = r.w;
                rj["height"] = r.h;
                j["rois"].push_back(rj);
            }
            break;
        case RoiSourceType::PerImageFile:
            j["roi_source"] = "per_image_file";
            j["per_image_rois_dir"] = perImageRoisDir;
            break;
        case RoiSourceType::PerImageList:
            j["roi_source"] = "per_image_list";
            j["per_image_rois_file"] = perImageRoisFile;
            break;
    }

    j["train"]["coreset_fraction"] = coresetFraction;
    j["train"]["max_features"] = maxFeatures;
    j["inference"]["anomaly_threshold"] = anomalyThreshold;
    j["model_dir"] = modelDir;

    return j.dump(2);
}

} // namespace aicore
