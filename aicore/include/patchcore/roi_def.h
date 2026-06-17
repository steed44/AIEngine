// ============================================================
// roi_def.h — 多 ROI 多模型支持的核心数据结构
// 功能：定义 ROI 矩形区域、多 ROI 全局配置、JSON 序列化
// 适用场景：一张大图包含多个检测区域，每个区域独立训练
//           PatchCore 模型并独立推理
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
// MultiRoiConfig — 多 ROI 完整配置
// 包含 backbone 公共参数 + ROI 列表 + 训练/推理参数
// 对应 JSON 文件格式，通过 FromJson / ToJson 序列化
// -------------------------------------------------------
struct MultiRoiConfig {
    // backbone 公共参数（所有 ROI 共享同一 backbone）
    std::string backboneType = "opencv_dnn";   // opencv_dnn / model_backend / libtorch
    std::string backendType = "onnxruntime";   // model_backend 时指定具体后端：onnxruntime / tensorrt / libtorch
    std::string backboneModelPath;             // .pt（libtorch）或 .onnx（opencv_dnn / model_backend）
    std::string backboneLayers = "layer2,layer3";
    int inputSize = 224;

    // ROI 区域列表
    std::vector<RoiDef> rois;

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
