#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

class AICORE_OPTIMIZER_API PythonEmbedding {
public:
    PythonEmbedding();
    ~PythonEmbedding();

    Status Initialize();
    Status RunScript(const std::string& script, const std::string& configJson,
                     std::string& output);
    void Finalize();

private:
    bool initialized_ = false;
};

} // namespace aicore
