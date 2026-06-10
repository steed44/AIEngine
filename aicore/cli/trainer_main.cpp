#include "trainer/trainer_api.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: AICoreTrainer <config.json>" << std::endl;
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Cannot open: " << argv[1] << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << file.rdbuf();

    std::string json = ss.str();
    const char* err = nullptr;
    int ret = aicore_train_run(json.c_str(), &err);
    if (ret != 0) {
        std::cerr << "Training failed: " << (err ? err : "unknown") << std::endl;
        return 1;
    }
    std::cout << "Training completed successfully." << std::endl;
    return 0;
}
