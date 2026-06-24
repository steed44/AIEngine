#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/cuda.h>
#include <torch/nn/utils/clip_grad.h>
#include <iostream>

int main() {
    try {
        // Tiny model: Conv2d(3, 16, 3)
        auto conv = torch::nn::Conv2d(3, 16, 3);
        auto input = torch::randn({2, 3, 32, 32});
        auto out = conv->forward(input);
        auto loss = out.mean();
        std::cout << "forward OK\n";
        
        loss.backward();
        std::cout << "backward OK\n";
        
        auto params = conv->parameters();
        torch::nn::utils::clip_grad_norm_(params, 10.0);
        std::cout << "clip_grad OK\n";
        
        torch::optim::SGD opt(params, 0.01);
        opt.step();
        std::cout << "step OK\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
