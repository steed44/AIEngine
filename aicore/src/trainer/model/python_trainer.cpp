#include "trainer/model/python_trainer.h"

namespace aicore {

PythonTrainer::PythonTrainer() {}

Status PythonTrainer::Train(const std::string& configJson) {
    std::string output;
    return py_.RunScript("scripts/train_yolo.py", configJson, output);
}

std::string PythonTrainer::GetLastError() const { return lastError_; }

} // namespace aicore
