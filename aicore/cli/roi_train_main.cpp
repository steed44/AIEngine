// ============================================================
// 文件: cli/roi_train_main.cpp
// 用途: 多 ROI PatchCore 训练命令行工具
// 从大图文件夹训练多个 ROI 的 PatchCore 模型
//
// 用法:
//   RoiTrain --config rois.json --data ./normal_samples/
//            [--model-dir ./models/]
// ============================================================
#include "patchcore/roi_trainer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string configPath, dataFolder, modelDir;
    bool forceStream = false, forceNoStream = false;

    // ---- 1. 解析命令行参数 ----
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) configPath = argv[++i];
        else if (arg == "--data" && i + 1 < argc) dataFolder = argv[++i];
        else if (arg == "--model-dir" && i + 1 < argc) modelDir = argv[++i];
        else if (arg == "--stream") forceStream = true;
        else if (arg == "--no-stream") forceNoStream = true;
    }

    // ---- 2. 检查必需参数 ----
    if (configPath.empty() || dataFolder.empty()) {
        std::cerr << "Usage: RoiTrain --config <rois.json> --data <folder> [options]\n"
                  << "  --model-dir <dir>   Output directory for models (default: ./models)\n"
                  << "  --stream            Force streaming mode (one image at a time)\n"
                  << "  --no-stream         Force batch mode (load all images into memory)\n"
                  << "\n"
                  << "  By default, auto-detects large images (>100MB) and uses streaming.\n"
                  << "\n"
                  << "  The backbone type is specified in rois.json:\n"
                  << "    \"backbone\": {\"type\": \"opencv_dnn\",  ...}  # OpenCV DNN (.onnx)\n"
                  << "    \"backbone\": {\"type\": \"model_backend\", ...}  # Backend factory (onnxruntime/tensorrt/libtorch)\n"
                  << "    \"backbone\": {\"type\": \"libtorch\",    ...}  # LibTorch JIT (.pt)\n"
                  << "\n"
                  << "Examples:\n"
                  << "  RoiTrain --config rois.json --data ./big_images/\n"
                  << "  RoiTrain --config rois.json --data ./small_images/ --no-stream\n";
        return 1;
    }

    // ---- 3. 执行多 ROI 训练 ----
    std::cout << "Multi-ROI PatchCore Training\n"
              << "  Config: " << configPath << "\n"
              << "  Data:   " << dataFolder << "\n";

    aicore::RoiTrainer trainer;
    trainer.SetForceStream(forceStream);
    trainer.SetForceNoStream(forceNoStream);
    auto s = trainer.TrainAll(configPath, dataFolder);
    if (!s) {
        std::cerr << "Training failed: " << s.message << std::endl;
        return 1;
    }

    std::cout << "All ROI models trained successfully." << std::endl;
    return 0;
}
