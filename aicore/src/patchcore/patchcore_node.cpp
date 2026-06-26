// ============================================================
// patchcore_node.cpp — PatchCore 异常检测节点实现
//
// 本文件实现了 IProcessor 接口的 PatchCoreNode，是 PatchCore
// 异常检测算法的推理入口。
//
// 核心功能：
//   1. 多 backbone 支持：opencv_dnn (CPU) / libtorch (GPU)
//   2. 大图像分片处理：超过 maxTileSize 的图像自动切分
//   3. 多线程并行：ThreadPool 并行处理各分片
//   4. 多尺度推理：可选的图像金字塔（1.0/0.75/0.5）
//   5. Tile Cache：帧内重复 ROI 的特征缓存
//   6. GPU 自动降级：GPU 提取失败时自动回退 CPU
//
// 数据流：
//   输入 Frame (cv::Mat) → Backbone.Extract → MemoryBank.ComputeAnomalyMap
//   → 异常热力图 → 输出 Frame (anomaly_score, is_anomaly)
// ============================================================

#include "patchcore/patchcore_node.h"
#include "patchcore/scheduler.h"
#include "engine/thread_pool.h"
#include <opencv2/imgproc.hpp>
#include <future>

namespace aicore {

// ============================================================
// Init — 初始化 PatchCoreNode
// ============================================================
/**
 * 初始化 PatchCore 节点
 *
 * 从配置中读取以下参数：
 *   - model_path: backbone 模型路径（必填）
 *   - memory_bank_path: memory bank 文件路径（必填）
 *   - backbone: backbone 类型，默认 "opencv_dnn"
 *   - input_size: 输入尺寸，默认 224
 *   - anomaly_threshold: 异常阈值，默认 0.5
 *   - max_tile_size: 分片最大尺寸，默认 0（不分片）
 *   - multi_scale: 是否启用多尺度推理，默认 0
 *
 * 创建三个 backbone 实例：
 *   1. backbone_: 默认 backbone（opencv_dnn）
 *   2. cpuBackbone_: CPU 降级用 backbone（opencv_dnn）
 *   3. gpuBackbone_: GPU 加速 backbone（libtorch）
 *
 * 加载 memory bank 并提升到 GPU。
 */
Status PatchCoreNode::Init(const NodeConfig& config) {
    name_ = config.count("name") ? config.at("name") : "patchcore";

    if (config.find("model_path") == config.end()) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: model_path is required"};
    }
    if (config.find("memory_bank_path") == config.end()) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: memory_bank_path is required"};
    }

    // 创建默认 backbone
    auto bt = config.find("backbone");
    std::string backboneType = (bt != config.end()) ? bt->second : "opencv_dnn";
    backbone_ = CreateBackbone(backboneType, config);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone type: " + backboneType};
    }
    auto s = backbone_->Init(config);
    if (!s) return s;

    // 创建 CPU 降级用 backbone
    auto cpuCfg = config;
    cpuCfg["backbone"] = "opencv_dnn";
    cpuBackbone_ = CreateBackbone("opencv_dnn", cpuCfg);
    if (cpuBackbone_) cpuBackbone_->Init(cpuCfg);

    // 创建 GPU backbone
    auto gpuCfg = config;
    gpuCfg["backbone"] = "libtorch";
    gpuBackbone_ = CreateBackbone("libtorch", gpuCfg);
    if (gpuBackbone_) gpuBackbone_->Init(gpuCfg);

    // 加载 memory bank 并提升到 GPU
    auto mn = config.find("memory_bank_path");
    auto loadStatus = memoryBank_.Load(mn->second);
    if (!loadStatus) {
        return Status{StatusCode::ErrorModelLoad,
            "patchcore: cannot load memory bank: " + loadStatus.message};
    }
    memoryBank_.PromoteToGPU();

    // 读取可选配置参数
    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    auto at = config.find("anomaly_threshold");
    if (at != config.end()) anomalyThreshold_ = std::stof(at->second);

    auto ts = config.find("max_tile_size");
    if (ts != config.end()) maxTileSize_ = std::stoi(ts->second);

    auto ms = config.find("multi_scale");
    if (ms != config.end()) multiScale_ = std::stoi(ms->second);

    // 创建线程池，使用硬件并发数
    threadPool_ = std::make_unique<ThreadPool>(
        std::thread::hardware_concurrency());

    return Status{};
}

// ============================================================
// ProcessTile — 处理单个图像分片
// ============================================================
/**
 * 处理单个图像分片（tile），提取特征并计算异常热力图
 *
 * 流程：
 *   1. 从原图中裁剪 ROI 区域
 *   2. 选择最佳 backbone（GPU → CPU → 默认）
 *   3. 提取 patch 级特征
 *   4. MemoryBank 最近邻搜索，计算异常距离图
 *   5. 缓存到 tileCache_，避免重复计算
 *
 * @param img 完整图像
 * @param roi ROI 区域坐标
 * @param tileMapOut 输出的异常热力图（CV_32F）
 */
Status PatchCoreNode::ProcessTile(const cv::Mat& img, const cv::Rect& roi,
                                   cv::Mat& tileMapOut) {
    cv::Mat tile = img(roi);

    // 选择最佳 backbone：优先 GPU，失败则降级到 CPU
    auto pickBackbone = [&]() -> IBackbone* {
        if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
            try {
                gpuBackbone_->Extract(tile);
                return gpuBackbone_.get();
            } catch (const std::runtime_error&) {
                // 静默降级：GPU 提取失败时回退 CPU backbone
            }
        }
        if (cpuBackbone_) return cpuBackbone_.get();
        return backbone_.get();
    };

    auto backbone = pickBackbone();
    auto features = backbone->Extract(tile);
    if (features.empty()) {
        return Status{StatusCode::ErrorModelInfer,
            "patchcore: backbone returned no features"};
    }

    auto anomalyData = memoryBank_.ComputeAnomalyMap(features, tile.rows, tile.cols);
    if (anomalyData.empty()) {
        return Status{StatusCode::ErrorInternal,
            "patchcore: anomaly map empty"};
    }

    // 缓存到 tileCache_，key 为 ROI 坐标
    TileKey key{roi.x, roi.y, roi.width, roi.height};
    tileMapOut = cv::Mat(tile.rows, tile.cols, CV_32F, anomalyData.data()).clone();
    tileCache_[key] = tileMapOut.clone();
    return Status{};
}

// ============================================================
// Process — 处理完整图像
// ============================================================
/**
 * 处理完整图像，输出异常热力图和异常得分
 *
 * 处理策略（根据配置选择）：
 *   1. 不分片 + 单尺度：直接全图推理（最快）
 *   2. 分片 + 无多尺度：将大图像切分为 tile，并行处理
 *   3. 不分片 + 多尺度：图像金字塔 1.0/0.75/0.5，max-fusion
 *   4. 分片 + 多尺度：分片 + 图像金字塔，最慢但最准确
 *
 * 输出 Frame 包含：
 *   - image: CV_32F 异常热力图
 *   - roiMap["anomaly_score"]: 最大异常距离
 *   - roiMap["is_anomaly"]: 是否超过阈值（0 或 1）
 */
Status PatchCoreNode::Process(const std::vector<Frame>& inputs,
                               std::vector<Frame>& outputs) {
    if (inputs.empty()) {
        return Status{StatusCode::ErrorInvalidInput, "patchcore: no input"};
    }

    cv::Mat img = inputs[0].image;
    if (img.empty()) {
        return Status{StatusCode::ErrorPreprocess, "patchcore: empty image"};
    }

    cv::Mat fullMap;
    // 判断是否需要分片处理
    bool needsTiling = (maxTileSize_ > 0) &&
        (img.cols > maxTileSize_ || img.rows > maxTileSize_);

    // 策略1：不分片 + 单尺度 — 直接全图推理
    if (!needsTiling && multiScale_ == 0) {
        auto pickBackbone = [&]() -> IBackbone* {
            if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
                try {
                    gpuBackbone_->Extract(img);
                    return gpuBackbone_.get();
                } catch (const std::runtime_error&) {
                    // 静默降级：GPU 提取失败时回退 CPU backbone
                }
            }
            if (cpuBackbone_) return cpuBackbone_.get();
            return backbone_.get();
        };

        auto backbone = pickBackbone();
        auto features = backbone->Extract(img);
        if (features.empty()) {
            return Status{StatusCode::ErrorModelInfer,
                "patchcore: backbone returned no features"};
        }
        auto anomalyData = memoryBank_.ComputeAnomalyMap(features, img.rows, img.cols);
        if (anomalyData.empty()) {
            return Status{StatusCode::ErrorInternal,
                "patchcore: anomaly map empty"};
        }
        fullMap = cv::Mat(img.rows, img.cols, CV_32F, anomalyData.data()).clone();
    }
    // 策略2：分片处理（大图像）
    else if (needsTiling) {
        tileCache_.clear();
        fullMap = cv::Mat::zeros(img.rows, img.cols, CV_32F);

        int tileH = std::min(maxTileSize_, img.rows);
        int tileW = std::min(maxTileSize_, img.cols);
        int overlap = 32;  // 分片重叠区域（像素），避免边界效应

        // 生成分片列表（带重叠）
        struct TileJob {
            cv::Rect roi;
            int idx;
        };
        std::vector<TileJob> tiles;
        for (int y0 = 0; y0 < img.rows; y0 += tileH) {
            for (int x0 = 0; x0 < img.cols; x0 += tileW) {
                // 重叠区域：上下左右各扩展 overlap 像素
                int y1 = std::min(y0 + tileH + overlap, img.rows);
                int x1 = std::min(x0 + tileW + overlap, img.cols);
                int cropY0 = std::max(0, y0 - overlap);
                int cropX0 = std::max(0, x0 - overlap);
                tiles.push_back({{cropX0, cropY0, x1 - cropX0, y1 - cropY0},
                                 (int)tiles.size()});
            }
        }

        // 并行处理所有分片
        std::vector<std::future<Status>> futures;
        for (auto& t : tiles) {
            futures.push_back(threadPool_->Enqueue([this, &img, &fullMap, t]() -> Status {
                cv::Mat tileMap;
                auto st = ProcessTile(img, t.roi, tileMap);
                if (!st) return st;
                // max-fusion：取重叠区域的最大异常值
                cv::Mat dst = fullMap(t.roi);
                cv::max(dst, tileMap, dst);
                return Status{};
            }));
        }

        // 等待所有分片完成
        for (auto& f : futures) {
            auto st = f.get();
            if (!st) return st;
        }

        // 策略3：分片 + 多尺度
        if (multiScale_) {
            const float scales[] = {0.75f, 0.5f};
            for (float scale : scales) {
                cv::Mat scaled;
                cv::resize(img, scaled, cv::Size(), scale, scale,
                           cv::INTER_LINEAR);

                std::vector<std::future<Status>> scaleFutures;
                for (auto& t : tiles) {
                    // 缩放 ROI 坐标
                    cv::Rect scaledRoi{
                        (int)(t.roi.x * scale),
                        (int)(t.roi.y * scale),
                        (int)(t.roi.width * scale),
                        (int)(t.roi.height * scale)
                    };
                    if (scaledRoi.x + scaledRoi.width > scaled.cols)
                        scaledRoi.width = scaled.cols - scaledRoi.x;
                    if (scaledRoi.y + scaledRoi.height > scaled.rows)
                        scaledRoi.height = scaled.rows - scaledRoi.y;
                    if (scaledRoi.width <= 0 || scaledRoi.height <= 0)
                        continue;

                    scaleFutures.push_back(
                        threadPool_->Enqueue([this, &scaled, &fullMap, scaledRoi, &t]() -> Status {
                            cv::Mat tileMap;
                            auto st = ProcessTile(scaled, scaledRoi, tileMap);
                            if (!st) return st;
                            // 放大回原图尺寸后 max-fusion
                            cv::Mat up;
                            cv::resize(tileMap, up, cv::Size(t.roi.width, t.roi.height),
                                       cv::INTER_LINEAR);
                            cv::Mat dst = fullMap(t.roi);
                            cv::max(dst, up, dst);
                            return Status{};
                        }));
                }
                for (auto& f : scaleFutures) {
                    auto st = f.get();
                    if (!st) return st;
                }
            }
        }
    }
    // 策略4：不分片 + 多尺度（图像金字塔）
    else {
        tileCache_.clear();
        fullMap = cv::Mat::zeros(img.rows, img.cols, CV_32F);

        const float scales[] = {1.0f, 0.75f, 0.5f};
        for (float scale : scales) {
            cv::Mat src;
            if (scale < 1.0f) {
                cv::resize(img, src, cv::Size(), scale, scale, cv::INTER_LINEAR);
            } else {
                src = img;
            }

            auto pickBackbone = [&]() -> IBackbone* {
                if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
                    try {
                        gpuBackbone_->Extract(src);
                        return gpuBackbone_.get();
                    } catch (const std::runtime_error&) {
                        // 静默降级：GPU 提取失败时回退 CPU backbone
                    }
                }
                if (cpuBackbone_) return cpuBackbone_.get();
                return backbone_.get();
            };

            auto backbone = pickBackbone();
            auto features = backbone->Extract(src);
            if (features.empty()) continue;

            auto anomalyData = memoryBank_.ComputeAnomalyMap(
                features, src.rows, src.cols);
            if (anomalyData.empty()) continue;

            cv::Mat scaleMap(src.rows, src.cols, CV_32F, anomalyData.data());
            if (scale < 1.0f) {
                // 放大回原图尺寸后 max-fusion
                cv::Mat up;
                cv::resize(scaleMap, up, img.size(), 0, 0, cv::INTER_LINEAR);
                cv::max(fullMap, up, fullMap);
            } else {
                cv::max(fullMap, scaleMap, fullMap);
            }
        }
    }

    // 计算异常得分：取热力图中的最大值
    double maxVal = 0;
    cv::minMaxLoc(fullMap, nullptr, &maxVal);

    Frame out(fullMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));

    return Status{};
}

} // namespace aicore
