#pragma once
#include "core/types.h"
#include "core/frame.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace aicore {

using NodeConfig = std::unordered_map<std::string, std::string>;

class IProcessor {
public:
    virtual ~IProcessor() = default;

    virtual Status Init(const NodeConfig& config) = 0;
    virtual Status Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) = 0;
    virtual std::string GetName() const = 0;
    virtual std::string GetType() const = 0;
};

} // namespace aicore
