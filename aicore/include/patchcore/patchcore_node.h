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

/**
 * PatchCore 异常检测节点
 *
 * 实现 PatchCore (WACV 2022) 算法的推理部分：
 *   1. 通过 backbone 提取图像的 patch 级特征
 *   2. 在 memory bank 中进行最近邻搜索
 *   3. 计算每个 patch 的异常距离
 *   4. 输出异常热力图和异常得分
 *
 * 特性：
 *   - 多 backbone 支持：opencv_dnn (CPU) / libtorch (GPU)
 *   - 大图像自动分片：超过 maxTileSize_ 的图像切分为 tile
 *   - 多线程并行：ThreadPool 并行处理各 tile
 *   - 多尺度推理：图像金字塔 1.0/0.75/0.5，max-fusion
 *   - Tile Cache：帧内重复 ROI 避免重复计算
 *   - GPU 自动降级：GPU 提取失败时回退 CPU
 *
 * 输入 Frame：cv::Mat（RGB/BGR 图像）
 * 输出 Frame：cv::Mat（CV_32F 异常热力图）
 *            roiMap["anomaly_score"] = 最大异常距离
 *            roiMap["is_anomaly"] = 是否超过阈值
 *
 * 配置参数（NodeConfig）：
 *   - model_path: backbone 模型路径（必填）
 *   - memory_bank_path: memory bank .bin 路径（必填）
 *   - backbone: 默认 backbone 类型，默认 "opencv_dnn"
 *   - input_size: 输入尺寸，默认 224
 *   - anomaly_threshold: 异常判定阈值，默认 0.5
 *   - max_tile_size: 分片最大尺寸（0=不分片），默认 1024
 *   - multi_scale: 是否启用多尺度（0=禁用，1=启用），默认 0
 */
class PatchCoreNode : public IProcessor {
public:
    // IProcessor 接口实现
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "patchcore"; }

private:
    /**
     * TileKey — 分片唯一标识
     * 由 ROI 坐标 (x, y, w, h) 组成，用于 tileCache_ 的 key。
     */
    struct TileKey {
        int x, y, w, h;
        bool operator==(const TileKey& o) const {
            return x == o.x && y == o.y && w == o.w && h == o.h;
        }
    };

    /**
     * TileKeyHash — TileKey 的哈希函数
     * 将 4 个 int 的位组合成一个 size_t，用于 unordered_map。
     */
    struct TileKeyHash {
        size_t operator()(const TileKey& k) const {
            return ((size_t)k.x << 0) ^ ((size_t)k.y << 16) ^ ((size_t)k.w << 32) ^ ((size_t)k.h << 48);
        }
    };

    /**
     * 处理单个图像分片
     * @param img 完整图像
     * @param roi ROI 区域坐标
     * @param tileMapOut 输出的异常热力图
     */
    Status ProcessTile(const cv::Mat& img, const cv::Rect& roi,
                       cv::Mat& tileMapOut);

    /** Tile 缓存：key=ROI 坐标，value=异常热力图 */
    using TileCache = std::unordered_map<TileKey, cv::Mat, TileKeyHash>;

    // 成员变量
    std::string name_;                              // 节点名称
    std::unique_ptr<IBackbone> backbone_;           // 默认 backbone
    std::unique_ptr<IBackbone> gpuBackbone_;        // GPU backbone (libtorch)
    std::unique_ptr<IBackbone> cpuBackbone_;        // CPU 降级 backbone (opencv_dnn)
    TieredMemoryBank memoryBank_;                   // 特征存储 + 最近邻搜索
    std::unique_ptr<ThreadPool> threadPool_;        // 线程池
    int inputSize_ = 224;                           // backbone 输入尺寸
    float anomalyThreshold_ = 0.5f;                 // 异常判定阈值
    int maxTileSize_ = 1024;                        // 分片最大尺寸（0=不分片）
    int multiScale_ = 0;                            // 多尺度推理开关
    TileCache tileCache_;                           // tile 缓存
};

} // namespace aicore
