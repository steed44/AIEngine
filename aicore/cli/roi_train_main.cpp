// ============================================================
// 文件: cli/roi_train_main.cpp
// 用途: 多 ROI PatchCore 训练命令行工具
// 从大图文件夹训练多个 ROI 的 PatchCore 模型
//
// 用法:
//   // 固定 ROI 坐标模式（向后兼容）:
//   RoiTrain --config rois.json --data ./normal_samples/
//
//   // 每图 ROI 模式（新）:
//   RoiTrain --config rois.json --data ./images/ --per-image-rois ./roi_annotations/
// ============================================================
#include "patchcore/roi_trainer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string configPath, dataFolder, modelDir, perImageRoisDir;
    bool forceStream = false, forceNoStream = false;
    bool perImageMode = false;

    // ---- 1. 解析命令行参数 ----
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) configPath = argv[++i];
        else if (arg == "--data" && i + 1 < argc) dataFolder = argv[++i];
        else if (arg == "--model-dir" && i + 1 < argc) modelDir = argv[++i];
        else if (arg == "--stream") forceStream = true;
        else if (arg == "--no-stream") forceNoStream = true;
        else if (arg == "--per-image-rois" && i + 1 < argc) {
            perImageRoisDir = argv[++i];
            perImageMode = true;
        }
    }

    // ---- 2. 检查必需参数 ----
    if (configPath.empty() || dataFolder.empty()) {
        std::cerr << "Usage: RoiTrain [options]\n"
                  << "\nOptions:\n"
                  << "  --config <file>        Config file (required)\n"
                  << "  --data <folder>        Training images folder (required)\n"
                  << "  --model-dir <dir>      Output directory for models (default: ./models)\n"
                  << "  --stream               Force streaming mode (one image at a time)\n"
                  << "  --no-stream            Force batch mode (load all images into memory)\n"
                  << "  --per-image-rois <dir> Per-image ROI mode: directory containing ROI JSONs\n"
                  << "                         Each image a.png uses a.json for ROI coordinates\n"
                  << "\nROI Modes:\n"
                  << "  Fixed (default): ROI coordinates from config file, same for all images\n"
                  << "  Per-image:       Each image has its own ROI coordinates, grouped by id\n"
                  << "\nExamples:\n"
                  << "  // Fixed ROI mode:\n"
                  << "  RoiTrain --config rois.json --data ./big_images/\n"
                  << "\n"
                  << "  // Per-image ROI mode (每张图有独立 ROI 坐标):\n"
                  << "  RoiTrain --config rois.json --data ./images/ --per-image-rois ./annotations/\n";
        return 1;
    }

    // ---- 3. 执行训练 ----
    std::cout << "Multi-ROI PatchCore Training\n"
              << "  Config: " << configPath << "\n"
              << "  Data:   " << dataFolder << "\n";

    aicore::RoiTrainer trainer;
    trainer.SetForceStream(forceStream);
    trainer.SetForceNoStream(forceNoStream);

    aicore::Status s;
    if (perImageMode) {
        if (perImageRoisDir.empty()) {
            std::cerr << "Error: --per-image-rois requires a directory path" << std::endl;
            return 1;
        }
        std::cout << "  Mode: Per-image ROI (coordinates from JSON files)\n"
                  << "  ROI dir: " << perImageRoisDir << "\n";
        s = trainer.TrainAllPerImage(configPath, dataFolder, perImageRoisDir);
    } else {
        std::cout << "  Mode: Fixed ROI (coordinates from config)\n";
        s = trainer.TrainAll(configPath, dataFolder);
    }

    if (!s) {
        std::cerr << "Training failed: " << s.message << std::endl;
        return 1;
    }

    std::cout << "All ROI models trained successfully." << std::endl;
    return 0;
}
