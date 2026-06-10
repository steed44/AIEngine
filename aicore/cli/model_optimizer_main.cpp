#include "optimizer/model_optimizer.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ModelOptimizer <config.json>" << std::endl;
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Cannot open: " << argv[1] << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << file.rdbuf();

    aicore::ModelOptimizer optimizer;
    auto s = optimizer.Optimize(ss.str());
    if (!s) {
        std::cerr << "Optimization failed: " << s.message << std::endl;
        return 1;
    }
    std::cout << "Optimization completed successfully." << std::endl;
    return 0;
}
