#include "patchcore/patchcore_trainer.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string dataPath, modelPath, outputPath = "memory_bank.bin";
    double coresetFrac = 0.1;
    int inputSize = 224;
    std::string layers = "layer2,layer3";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--input_size" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) layers = argv[++i];
        else if (arg == "--coreset" && i + 1 < argc) coresetFrac = std::stod(argv[++i]);
    }

    if (dataPath.empty() || modelPath.empty()) {
        std::cerr << "Usage: PatchCoreTrain --data <folder> --model <backbone.onnx> [options]\n"
                  << "  --output <path>    Memory bank output path (default: memory_bank.bin)\n"
                  << "  --input_size <n>   Input image size (default: 224)\n"
                  << "  --layers <names>   Backbone layer names (default: layer2,layer3)\n"
                  << "  --coreset <frac>   Coreset fraction (default: 0.1)\n";
        return 1;
    }

    aicore::PatchCoreTrainConfig cfg;
    cfg.inputSize = inputSize;
    cfg.backboneLayers = layers;
    cfg.coresetFraction = coresetFrac;

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
