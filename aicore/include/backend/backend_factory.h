#pragma once
#include "core/model_backend.h"
#include <memory>

namespace aicore {

class BackendFactory {
public:
    static std::unique_ptr<IModelBackend> Create(BackendType type);
};

} // namespace aicore
