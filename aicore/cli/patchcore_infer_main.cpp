#include "patchcore/patchcore_node.h"
#include "patchcore/patchcore_visualize.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string modelPath, bankPath, imagePath, outputPath = "result.png", backbone = "opencv_dnn";
    int inputSize = 224;
    float threshold = 0.5f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--memory" && i + 1 < argc) bankPath = argv[++i];
        else if (arg == "--image" && i + 1 < argc) imagePath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--backbone" && i + 1 < argc) backbone = argv[++i];
        else if (arg == "--input_size" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
    }

    if (modelPath.empty() || bankPath.empty() || imagePath.empty()) {
        std::cerr << "Usage: PatchCoreInfer --model <model> --memory <bank.bin> --image <img> [options]\n"
                  << "  --backbone <t>    Backbone type: opencv_dnn (default) | libtorch\n"
                  << "  --output <path>   Output image (default: result.png)\n"
                  << "  --input_size <n>  Input size (default: 224)\n"
                  << "  --threshold <f>   Anomaly threshold (default: 0.5)\n"
                  << "Example:\n"
                  << "  PatchCoreInfer --model backbone.onnx --memory bank.bin --image test.png\n";
        return 1;
    }

    aicore::NodeConfig cfg;
    cfg["model_path"] = modelPath;
    cfg["memory_bank_path"] = bankPath;
    cfg["backbone"] = backbone;
    cfg["input_size"] = std::to_string(inputSize);
    cfg["anomaly_threshold"] = std::to_string(threshold);

    aicore::PatchCoreNode node;
    auto s = node.Init(cfg);
    if (!s) {
        std::cerr << "Init failed: " << s.message << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    aicore::Frame input(img, 1);
    std::vector<aicore::Frame> outputs;
    s = node.Process({input}, outputs);
    if (!s) {
        std::cerr << "Inference failed: " << s.message << std::endl;
        return 1;
    }

    auto& result = outputs[0];
    float score = result.roiMap.count("anomaly_score") ? result.roiMap["anomaly_score"] : 0;
    bool isAnomaly = result.roiMap.count("is_anomaly") ? result.roiMap["is_anomaly"] > 0 : false;

    std::cout << "Anomaly score: " << score << "\n"
              << "Is anomaly:    " << (isAnomaly ? "YES" : "no") << "\n";

    cv::Mat overlay = aicore::DrawAnomalyOverlay(result.image, img, 0.6f, 0.1f);
    cv::imwrite(outputPath, overlay);
    std::cout << "Result saved to " << outputPath << std::endl;

    return 0;
}
