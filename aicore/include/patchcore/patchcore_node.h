#pragma once
#include "core/processor.h"
#include "patchcore/tiered_memory_bank.h"
#include "patchcore/backbone.h"
#include "patchcore/faiss_index_bridge.h"
#include <string>
#include <vector>
#include <memory>
#include "engine/thread_pool.h"

namespace aicore {

/**
 * PatchCore 异常检测节点
 *
 * 实现 PatchCore (WACV 2022) 算法的推理部分：
 *   1. 通过 backbone 提取图像的 patch 级特征
 *   2. 在 memory bank 中进行最近邻搜索（暴力或 FAISS 近似）
 *   3. 计算每个 patch 的异常距离
 *   4. 输出异常热力图和异常得分
 *
 * 特性：
 *   - 多 backbone 支持：opencv_dnn (CPU) / libtorch (GPU)
 *   - 大图像自动分片：超过 maxTileSize_ 的图像切分为 tile
 *   - 多线程并行：ThreadPool 并行处理各 tile
 *   - 多尺度推理：图像金字塔 1.0/0.75/0.5，max-fusion
 *   - GPU 自动降级：GPU 提取失败时回退 CPU
 *   - FAISS 近似最近邻搜索：IVF/HNSW/BruteForce 三种算法可选
 *   - FAISS 自动降级：索引文件损坏或构建失败时回退暴力搜索
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
 *   - search_algorithm: FAISS 算法（"brute_force"/"ivf"/"hnsw"），默认 "brute_force"
 *   - faiss_nlist: IVF 聚类中心数，默认 100
 *   - faiss_nprobe: IVF 探查簇数，默认 16
 *   - faiss_m: HNSW 连接数，默认 16
 *   - faiss_ef_construction: HNSW 构建动态列表，默认 200
 *   - faiss_ef_search: HNSW 搜索动态列表，默认 64
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
     * 处理单个图像分片
     * @param img 完整图像
     * @param roi ROI 区域坐标
     * @param tileMapOut 输出的异常热力图
     */
    Status ProcessTile(const cv::Mat& img, const cv::Rect& roi,
                       cv::Mat& tileMapOut);

    // 搜索分发：暴力搜索走 memoryBank_，FAISS 模式走 faissBridge_
    cv::Mat DispatchAnomalyMap(const std::vector<PatchFeature>& features,
                                int rows, int cols);

    // 成员变量
    std::string name_;                              // 节点名称
    std::unique_ptr<IBackbone> backbone_;           // 默认 backbone（创建时指定的类型）
    std::unique_ptr<IBackbone> gpuBackbone_;        // GPU backbone（libtorch），用于加速推理
    std::unique_ptr<IBackbone> cpuBackbone_;        // CPU 降级 backbone（opencv_dnn），GPU 失败时回退
    TieredMemoryBank memoryBank_;                   // 三级特征存储 + 最近邻搜索（热/温/冷）
    std::unique_ptr<ThreadPool> threadPool_;        // 线程池，并行处理 tile 和多尺度
    int inputSize_ = 224;                           // backbone 输入尺寸（224×224）
    float anomalyThreshold_ = 0.5f;                 // 异常判定阈值（>阈值则标记为异常）
    int maxTileSize_ = 1024;                        // 分片最大尺寸（0=不分片）
    int multiScale_ = 0;                            // 多尺度推理开关（1=启用图像金字塔）
    // FAISS 近似最近邻搜索（可选，默认 BruteForce）
    std::unique_ptr<FaissIndexBridge> faissBridge_; // FAISS 桥接层
    FaissSearchAlgorithm searchAlgo_ = FaissSearchAlgorithm::BruteForce;
    int faissNlist_ = 100;
    int faissNprobe_ = 16;
    int faissM_ = 16;
    int faissEfConstruction_ = 200;
    int faissEfSearch_ = 64;
};

} // namespace aicore
