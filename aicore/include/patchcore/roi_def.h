// ============================================================
// roi_def.h — 多 ROI 多模型支持的核心数据结构
// 功能：定义 ROI 矩形区域、多 ROI 全局配置、JSON 序列化
// 适用场景：一张大图包含多个检测区域，每个区域独立训练
//           PatchCore 模型并独立推理
//
// 两种模式：
//   1. 固定 ROI 模式（现有）：配置文件中定义固定坐标，所有训练图
//      使用相同 ROI 位置
//   2. 每图 ROI 模式（新增）：每张训练图有自己的 ROI 坐标列表，
//      通过 per_image_rois.json 提供，相同编号的 ROI 跨图合并
//      训练同一个模型
// ============================================================
#pragma once
#include "core/types.h"
#include <string>
#include <vector>
#include <fstream>

namespace aicore {

// -------------------------------------------------------
// RoiDef — 单个感兴趣区域（Region of Interest）定义
// 字段：
//   id   — ROI 唯一标识，也是模型文件名（如 "connector" → connector.bin）
//   x, y, w, h — 大图中的矩形区域（像素坐标）
// 典型使用：PCB 检测中 "芯片引脚区"、"焊点区" 各为一个 ROI
// -------------------------------------------------------
struct RoiDef {
    std::string id;
    int x = 0, y = 0, w = 0, h = 0;
};

// -------------------------------------------------------
// PerImageRoiConfig — 单张大图的 ROI 坐标配置
// 用于"每图 ROI"模式：每张训练图有自己的 ROI 列表
// JSON 格式：
// {
//   "image": "train_img_001.png",
//   "rois": [
//     {"id": "1", "x": 10, "y": 20, "width": 120, "height": 80},
//     {"id": "2", "x": 200, "y": 150, "width": 100, "height": 100},
//     ...
//   ]
// }
// id 为字符串编号（如 "1"~"10"），相同 id 的 ROI 跨图合并训练
// -------------------------------------------------------
struct PerImageRoiConfig {
    std::string imagePath;   // 图片文件名（不含路径）或完整路径
    std::vector<RoiDef> rois;

    // 从 JSON 文件加载单张图的 ROI 配置
    static Status FromJson(const std::string& path, PerImageRoiConfig& out);

    // 序列化为 JSON 字符串
    std::string ToJson() const;
};

// -------------------------------------------------------
// MultiRoiConfig — 多 ROI 完整配置
// 包含 backbone 公共参数 + ROI 列表 + 训练/推理参数
// 对应 JSON 文件格式，通过 FromJson / ToJson 序列化
// -------------------------------------------------------
struct MultiRoiConfig {
    // ROI 数据来源模式
    enum class RoiSourceType {
        Fixed,            // 固定坐标：所有图使用同一组 ROI 坐标
        PerImageFile,     // 每图独立文件：每张图对应一个 JSON 文件
        PerImageList      // 单文件列表：一个 JSON 包含所有图的 ROI
    };

    RoiSourceType roiSource = RoiSourceType::Fixed;

    // backbone 公共参数（所有 ROI 共享同一 backbone）
    std::string backboneType = "opencv_dnn";   // opencv_dnn / model_backend / libtorch
    std::string backendType = "onnxruntime";   // model_backend 时指定具体后端：onnxruntime / tensorrt / libtorch
    std::string backboneModelPath;             // .pt（libtorch）或 .onnx（opencv_dnn / model_backend）
    std::string backboneLayers = "layer2,layer3";
    int inputSize = 224;

    // ROI 区域列表（固定坐标模式使用）
    std::vector<RoiDef> rois;

    // 每图 ROI 模式使用
    std::string perImageRoisDir;       // 存放 per-image ROI JSON 的目录
    std::string perImageRoisFile;      // 单文件模式：包含所有图 ROI 的 JSON 路径
    std::vector<PerImageRoiConfig> perImageConfigs;  // 解析后的每图配置列表

    // 训练参数
    double coresetFraction = 0.1;
    size_t maxFeatures = 100000;

    // 推理参数
    float anomalyThreshold = 0.5f;

    // ROI 模型目录（训练输出 / 推理输入）
    std::string modelDir = "./models";

    // 从 JSON 文件加载配置
    // @param path JSON 配置文件路径
    // @return 解析成功返回 Status{}，否则返回错误
    Status FromJson(const std::string& path);

    // 序列化为 JSON 字符串（用于调试）
    std::string ToJson() const;
};

} // namespace aicore
