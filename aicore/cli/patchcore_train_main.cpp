// ============================================================
// 文件: cli/patchcore_train_main.cpp
// 用途: PatchCore 训练命令行工具入口
//   从图片文件夹提取特征 → 构建存入 memory_bank.bin 的记忆库
// ============================================================

#include "patchcore/patchcore_trainer.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string dataPath, modelPath, outputPath = "memory_bank.bin";
    double coresetFrac = 0.1;
    int inputSize = 224;
    std::string layers = "layer2,layer3";
    std::string backbone = "opencv_dnn";

    // ---- 1. 解析命令行参数 ----
    // 参数说明：
    //   --data <folder>        正常样本图片文件夹路径（必需）
    //   --model <file>         backbone 模型文件路径（必需）
    //                           libtorch 模式需要 .pt TorchScript 文件
    //                           opencv_dnn 模式需要 .onnx 文件
    //   --backbone <type>      backbone 类型：opencv_dnn（默认）| libtorch
    //   --output <path>        MemoryBank 输出路径（默认: memory_bank.bin）
    //   --input_size <n>       输入图像缩放尺寸（默认: 224）
    //   --layers <names>       backbone 待提取的中间层名，逗号分隔（默认: layer2,layer3）
    //                          仅 libtorch 模式需要（.pt 模型已固定输出层时仍可省略）
    //   --coreset <frac>       Coreset 采样比例（默认: 0.1）
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--backbone" && i + 1 < argc) backbone = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--input_size" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) layers = argv[++i];
        else if (arg == "--coreset" && i + 1 < argc) coresetFrac = std::stod(argv[++i]);
    }

    // ---- 2. 检查必需参数 ----
    if (dataPath.empty() || modelPath.empty()) {
        std::cerr << "Usage: PatchCoreTrain --data <folder> --model <model> [options]\n"
                  << "  --backbone <type>  Backbone type: opencv_dnn (default) | libtorch\n"
                  << "  --output <path>    Memory bank output path (default: memory_bank.bin)\n"
                  << "  --input_size <n>   Input image size (default: 224)\n"
                  << "  --layers <names>   Backbone layer names (default: layer2,layer3)\n"
                  << "  --coreset <frac>   Coreset fraction (default: 0.1)\n"
                  << "Examples:\n"
                  << "  PatchCoreTrain --data ./good --model backbone.onnx\n"
                  << "  PatchCoreTrain --data ./good --model backbone.pt --backbone libtorch\n";
        return 1;
    }

    // ---- 3. 校验 backbone 类型 ----
    if (backbone != "opencv_dnn" && backbone != "libtorch") {
        std::cerr << "Error: unknown backbone type '" << backbone
                  << "'. Supported: opencv_dnn, libtorch\n";
        return 1;
    }

    // ---- 4. 配置训练参数 ----
    aicore::PatchCoreTrainConfig cfg;
    cfg.inputSize = inputSize;
    cfg.backboneLayers = layers;
    cfg.backboneType = backbone;  // 传递 backbone 类型给训练器
    cfg.coresetFraction = coresetFrac;

    // ---- 5. 执行训练 ----
    aicore::PatchCoreTrainer trainer;
    auto s = trainer.TrainFromFolder(dataPath, modelPath, outputPath, cfg);
    if (!s) {
        std::cerr << "Training failed: " << s.message << std::endl;
        return 1;
    }
    std::cout << "Memory bank saved to " << outputPath << std::endl;
    std::cout << "Total images processed." << std::endl;
    return 0;
}
