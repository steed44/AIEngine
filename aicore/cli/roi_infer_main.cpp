// ============================================================
// 文件: cli/roi_infer_main.cpp
// 用途: 多 ROI PatchCore 推理命令行工具
// 对单张大图执行多 ROI 异常检测，输出标注结果图
//
// 用法:
//   // 固定 ROI 坐标模式:
//   RoiInfer --config rois.json --image test.png --output result.png
//
//   // 每图 ROI 模式:
//   RoiInfer --config rois.json --image test.png --per-image-rois ./roi_annotations/ --output result.png
// ============================================================
#include "patchcore/multi_roi_node.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string configPath, imagePath, outputPath = "result.png";
    std::string perImageRoisDir;
    bool perImageMode = false;

    // ---- 1. 解析命令行参数 ----
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) configPath = argv[++i];
        else if (arg == "--image" && i + 1 < argc) imagePath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--per-image-rois" && i + 1 < argc) {
            perImageRoisDir = argv[++i];
            perImageMode = true;
        }
    }

    // ---- 2. 检查必需参数 ----
    if (configPath.empty() || imagePath.empty()) {
        std::cerr << "Usage: RoiInfer [options]\n"
                  << "\nOptions:\n"
                  << "  --config <file>         Config file (required)\n"
                  << "  --image <file>          Input image (required)\n"
                  << "  --output <file>         Output image (default: result.png)\n"
                  << "  --per-image-rois <dir>  Per-image ROI mode: directory containing ROI JSONs\n"
                  << "\nExamples:\n"
                  << "  // Fixed ROI mode:\n"
                  << "  RoiInfer --config rois.json --image test.png --output result.png\n"
                  << "\n"
                  << "  // Per-image ROI mode:\n"
                  << "  RoiInfer --config rois.json --image test.png --per-image-rois ./annotations/ --output result.png\n";
        return 1;
    }

    // ---- 3. 初始化多 ROI 推理节点 ----
    aicore::MultiRoiNode node;
    aicore::NodeConfig nodeCfg;
    nodeCfg["config_path"] = configPath;
    nodeCfg["draw_overlay"] = "true";

    auto s = node.Init(nodeCfg);
    if (!s) {
        std::cerr << "Init failed: " << s.message << std::endl;
        return 1;
    }

    // ---- 4. 加载测试图像 ----
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    // ---- 5. 执行推理 ----
    aicore::Frame input(img, 1);
    std::vector<aicore::Frame> inputs = {input};
    std::vector<aicore::Frame> outputs;

    if (perImageMode) {
        // 每图模式：动态加载 ROI 坐标
        std::string imageName = imagePath;
        s = node.LoadPerImageRois(imageName, perImageRoisDir);
        if (!s) {
            std::cerr << "LoadPerImageRois failed: " << s.message << std::endl;
            return 1;
        }

        s = node.Process(inputs, outputs);
    } else {
        // 固定模式：直接使用初始化时的 ROI 配置
        s = node.Process(inputs, outputs);
    }

    if (!s) {
        std::cerr << "Inference failed: " << s.message << std::endl;
        return 1;
    }

    // ---- 6. 输出结果 ----
    if (outputs.empty()) {
        std::cerr << "No output from inference" << std::endl;
        return 1;
    }

    const auto& result = outputs[0];

    // 保存标注图
    cv::imwrite(outputPath, result.image);
    std::cout << "Result saved to " << outputPath << std::endl;

    // 打印各 ROI 得分
    std::cout << "\nROI Results:\n";
    for (auto& [key, val] : result.roiMap) {
        std::cout << "  " << key << " = " << val << std::endl;
    }

    // 显示结果窗口（如果环境支持 GUI）
    try {
        cv::imshow("Multi-ROI Result", result.image);
        cv::waitKey(0);
    } catch (...) {
        // 无 GUI 环境时静默跳过
    }

    return 0;
}
