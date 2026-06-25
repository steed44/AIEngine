#pragma once
#include "core/processor.h"
#include "patchcore/tiered_memory_bank.h"
#include "patchcore/backbone.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "engine/thread_pool.h"

namespace aicore {

class PatchCoreNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "patchcore"; }

private:
    struct TileKey {
        int x, y, w, h;
        bool operator==(const TileKey& o) const {
            return x == o.x && y == o.y && w == o.w && h == o.h;
        }
    };
    struct TileKeyHash {
        size_t operator()(const TileKey& k) const {
            return ((size_t)k.x << 0) ^ ((size_t)k.y << 16) ^ ((size_t)k.w << 32) ^ ((size_t)k.h << 48);
        }
    };

    Status ProcessTile(const cv::Mat& img, const cv::Rect& roi,
                       cv::Mat& tileMapOut);
    using TileCache = std::unordered_map<TileKey, cv::Mat, TileKeyHash>;

    std::string name_;
    std::unique_ptr<IBackbone> backbone_;
    std::unique_ptr<IBackbone> gpuBackbone_;
    std::unique_ptr<IBackbone> cpuBackbone_;
    TieredMemoryBank memoryBank_;
    std::unique_ptr<ThreadPool> threadPool_;
    int inputSize_ = 224;
    float anomalyThreshold_ = 0.5f;
    int maxTileSize_ = 1024;
    int multiScale_ = 0;       // 多尺度推理 (0=禁用, 1=启用)
    TileCache tileCache_;
};

} // namespace aicore
