#pragma once
#include "core/types.h"
#include "optimizer/python_embedding.h"
#include <string>

namespace aicore {

class PythonTrainer {
public:
    PythonTrainer();
    Status Train(const std::string& configJson);
    std::string GetLastError() const;

private:
    PythonEmbedding py_;
    std::string lastError_;
};

} // namespace aicore
