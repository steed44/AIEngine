#include "patchcore/patchcore_node.h"
#include "patchcore/scheduler.h"
#include "engine/thread_pool.h"
#include <opencv2/imgproc.hpp>
#include <future>

namespace aicore {

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

    auto bt = config.find("backbone");
    std::string backboneType = (bt != config.end()) ? bt->second : "opencv_dnn";

    backbone_ = CreateBackbone(backboneType, config);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone type: " + backboneType};
    }
    auto s = backbone_->Init(config);
    if (!s) return s;

    auto cpuCfg = config;
    cpuCfg["backbone"] = "opencv_dnn";
    cpuBackbone_ = CreateBackbone("opencv_dnn", cpuCfg);
    if (cpuBackbone_) cpuBackbone_->Init(cpuCfg);

    auto gpuCfg = config;
    gpuCfg["backbone"] = "libtorch";
    gpuBackbone_ = CreateBackbone("libtorch", gpuCfg);
    if (gpuBackbone_) gpuBackbone_->Init(gpuCfg);

    auto mn = config.find("memory_bank_path");
    auto loadStatus = memoryBank_.Load(mn->second);
    if (!loadStatus) {
        return Status{StatusCode::ErrorModelLoad,
            "patchcore: cannot load memory bank: " + loadStatus.message};
    }
    memoryBank_.PromoteToGPU();

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    auto at = config.find("anomaly_threshold");
    if (at != config.end()) anomalyThreshold_ = std::stof(at->second);

    auto ts = config.find("max_tile_size");
    if (ts != config.end()) maxTileSize_ = std::stoi(ts->second);

    auto ms = config.find("multi_scale");
    if (ms != config.end()) multiScale_ = std::stoi(ms->second);

    threadPool_ = std::make_unique<ThreadPool>(
        std::thread::hardware_concurrency());

    return Status{};
}

Status PatchCoreNode::ProcessTile(const cv::Mat& img, const cv::Rect& roi,
                                   cv::Mat& tileMapOut) {
    cv::Mat tile = img(roi);

    auto pickBackbone = [&]() -> IBackbone* {
        if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
            try {
                gpuBackbone_->Extract(tile);
                return gpuBackbone_.get();
            } catch (const std::runtime_error&) {
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

    TileKey key{roi.x, roi.y, roi.width, roi.height};
    tileMapOut = cv::Mat(tile.rows, tile.cols, CV_32F, anomalyData.data()).clone();
    tileCache_[key] = tileMapOut.clone();
    return Status{};
}

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
    bool needsTiling = (maxTileSize_ > 0) &&
        (img.cols > maxTileSize_ || img.rows > maxTileSize_);

    if (!needsTiling && multiScale_ == 0) {
        auto pickBackbone = [&]() -> IBackbone* {
            if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
                try {
                    gpuBackbone_->Extract(img);
                    return gpuBackbone_.get();
                } catch (const std::runtime_error&) {
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
    } else if (needsTiling) {
        tileCache_.clear();
        fullMap = cv::Mat::zeros(img.rows, img.cols, CV_32F);

        int tileH = std::min(maxTileSize_, img.rows);
        int tileW = std::min(maxTileSize_, img.cols);
        int overlap = 32;

        struct TileJob {
            cv::Rect roi;
            int idx;
        };
        std::vector<TileJob> tiles;
        for (int y0 = 0; y0 < img.rows; y0 += tileH) {
            for (int x0 = 0; x0 < img.cols; x0 += tileW) {
                int y1 = std::min(y0 + tileH + overlap, img.rows);
                int x1 = std::min(x0 + tileW + overlap, img.cols);
                int cropY0 = std::max(0, y0 - overlap);
                int cropX0 = std::max(0, x0 - overlap);
                tiles.push_back({{cropX0, cropY0, x1 - cropX0, y1 - cropY0},
                                 (int)tiles.size()});
            }
        }

        std::vector<std::future<Status>> futures;
        for (auto& t : tiles) {
            futures.push_back(threadPool_->Enqueue([this, &img, &fullMap, t]() -> Status {
                cv::Mat tileMap;
                auto st = ProcessTile(img, t.roi, tileMap);
                if (!st) return st;
                cv::Mat dst = fullMap(t.roi);
                cv::max(dst, tileMap, dst);
                return Status{};
            }));
        }

        for (auto& f : futures) {
            auto st = f.get();
            if (!st) return st;
        }

        if (multiScale_) {
            const float scales[] = {0.75f, 0.5f};
            for (float scale : scales) {
                cv::Mat scaled;
                cv::resize(img, scaled, cv::Size(), scale, scale,
                           cv::INTER_LINEAR);

                std::vector<std::future<Status>> scaleFutures;
                for (auto& t : tiles) {
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
    } else {
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
                    } catch (const std::runtime_error&) {}
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
                cv::Mat up;
                cv::resize(scaleMap, up, img.size(), 0, 0, cv::INTER_LINEAR);
                cv::max(fullMap, up, fullMap);
            } else {
                cv::max(fullMap, scaleMap, fullMap);
            }
        }
    }

    double maxVal = 0;
    cv::minMaxLoc(fullMap, nullptr, &maxVal);

    Frame out(fullMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));

    return Status{};
}

} // namespace aicore