#include "patchcore/roi_trainer.h"
#include "patchcore/patchcore_trainer.h"
#include "patchcore/folder_dataset.h"
#include "patchcore/backbone.h"
#include "patchcore/coreset_sampler.h"
#include "patchcore/memory_bank.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <random>
#include <iostream>
#include <unordered_map>

namespace aicore {

RoiTrainer::RoiTrainer()
    : forceStream_(false), forceNoStream_(false) {}

// -------------------------------------------------------
// 默认入口：根据图片大小自动选择流式或批量模式
// -------------------------------------------------------
Status RoiTrainer::TrainAll(const std::string& configPath,
                              const std::string& dataFolder) {
    if (forceStream_) {
        return TrainAllStreaming(configPath, dataFolder);
    }
    if (forceNoStream_) {
        return TrainAllBatch(configPath, dataFolder);
    }

    // 自动检测：取前 5 张图判断是否为大图
    auto files = FolderDataset::ListImageFiles(dataFolder);
    bool large = false;
    int checked = 0;
    for (auto& f : files) {
        if (IsLargeImage(f, 100)) { large = true; break; }
        if (++checked >= 5) break;
    }

    if (large) {
        std::cout << "[Auto] Large images detected, using streaming mode"
                  << std::endl;
        return TrainAllStreaming(configPath, dataFolder);
    }
    return TrainAllBatch(configPath, dataFolder);
}

// -------------------------------------------------------
// 流式模式：每张大图只读一次，共享 backbone，所有 ROI 同时处理
// -------------------------------------------------------
Status RoiTrainer::TrainAllStreaming(const std::string& configPath,
                                       const std::string& dataFolder) {
    // ---- 1. 加载配置 ----
    MultiRoiConfig cfg;
    auto s = cfg.FromJson(configPath);
    if (!s) { lastError_ = s.message; return s; }

    namespace fs = std::filesystem;
    fs::create_directories(cfg.modelDir);

    // ---- 2. 列出所有训练图路径（不加载） ----
    auto imagePaths = FolderDataset::ListImageFiles(dataFolder);
    if (imagePaths.empty()) {
        lastError_ = "no training images found in " + dataFolder;
        return Status{StatusCode::ErrorInvalidInput, lastError_};
    }
    std::cout << "Found " << imagePaths.size() << " training images (streaming)"
              << std::endl;

    // ---- 3. 初始化 backbone（共享，只一次） ----
    NodeConfig backboneConfig;
    backboneConfig["model_path"] = cfg.backboneModelPath;
    backboneConfig["backbone_layers"] = cfg.backboneLayers;
    backboneConfig["input_size"] = std::to_string(cfg.inputSize);
    if (cfg.backboneType == "model_backend") {
        backboneConfig["backend_type"] = cfg.backendType;
    }

    auto backbone = CreateBackbone(cfg.backboneType, backboneConfig);
    if (!backbone) {
        lastError_ = "unknown backbone: " + cfg.backboneType;
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
    s = backbone->Init(backboneConfig);
    if (!s) return s;

    // ---- 4. 为每个 ROI 准备特征累积容器 ----
    std::vector<std::vector<PatchFeature>> roiFeatures(cfg.rois.size());

    // ---- 5. 单次遍历磁盘 ----
    for (size_t fi = 0; fi < imagePaths.size(); fi++) {
        auto img = cv::imread(imagePaths[fi]);
        if (img.empty()) {
            std::cerr << "  Warning: cannot read " << imagePaths[fi]
                      << ", skipping" << std::endl;
            continue;
        }

        if ((fi + 1) % 50 == 0) {
            std::cout << "  Processed " << (fi + 1) << "/"
                      << imagePaths.size() << " images" << std::endl;
        }

        for (size_t ri = 0; ri < cfg.rois.size(); ri++) {
            const auto& roi = cfg.rois[ri];
            cv::Rect roiRect(roi.x, roi.y, roi.w, roi.h);

            // 检查 ROI 是否越界
            if (roiRect.x + roiRect.width > img.cols ||
                roiRect.y + roiRect.height > img.rows) {
                std::cerr << "  Warning: ROI " << roi.id
                          << " out of bounds on " << imagePaths[fi]
                          << ", skipping" << std::endl;
                continue;
            }

            cv::Mat roiCrop = img(roiRect).clone();
            auto feats = backbone->Extract(roiCrop);
            roiFeatures[ri].insert(roiFeatures[ri].end(),
                                   feats.begin(), feats.end());
        }
        // img 离开作用域，自动释放
    }

    // ---- 6. 对每个 ROI 执行 Coreset 采样 + 保存 ----
    PatchCoreTrainConfig trainCfg;
    trainCfg.inputSize = cfg.inputSize;
    trainCfg.backboneLayers = cfg.backboneLayers;
    trainCfg.backboneType = cfg.backboneType;
    trainCfg.backendType = cfg.backendType;
    trainCfg.coresetFraction = cfg.coresetFraction;
    trainCfg.maxFeatures = cfg.maxFeatures;

    for (size_t ri = 0; ri < cfg.rois.size(); ri++) {
        const auto& roi = cfg.rois[ri];
        std::cout << "[" << (ri + 1) << "/" << cfg.rois.size()
                  << "] Sampling ROI: " << roi.id
                  << " (" << roiFeatures[ri].size() << " features)"
                  << std::endl;

        if (roiFeatures[ri].empty()) {
            std::cerr << "  Warning: ROI " << roi.id
                      << " has no features, skipping" << std::endl;
            continue;
        }

        auto& feats = roiFeatures[ri];
        if (feats.size() > trainCfg.maxFeatures) {
            std::shuffle(feats.begin(), feats.end(),
                         std::mt19937(std::random_device()()));
            feats.resize(trainCfg.maxFeatures);
        }

        size_t targetSize = static_cast<size_t>(feats.size() *
                              trainCfg.coresetFraction);
        if (targetSize == 0) targetSize = 1;

        CoresetSampler sampler;
        auto indices = sampler.Sample(feats, targetSize);

        MemoryBank bank;
        std::vector<PatchFeature> coreFeatures;
        for (auto idx : indices) {
            coreFeatures.push_back(feats[idx]);
        }
        bank.Build(coreFeatures);

        std::string modelPath = cfg.modelDir + "/" + roi.id + ".bin";
        auto saveStatus = bank.Save(modelPath);
        if (!saveStatus) {
            lastError_ = saveStatus.message;
            return saveStatus;
        }

        std::cout << "  -> Saved to " << modelPath
                  << " (" << coreFeatures.size() << " features)"
                  << std::endl;
    }

    return Status{};
}

// -------------------------------------------------------
// 批量模式：一次性加载全部图片到内存（小图/旧行为）
// -------------------------------------------------------
Status RoiTrainer::TrainAllBatch(const std::string& configPath,
                                   const std::string& dataFolder) {
    MultiRoiConfig cfg;
    auto s = cfg.FromJson(configPath);
    if (!s) { lastError_ = s.message; return s; }

    namespace fs = std::filesystem;
    fs::create_directories(cfg.modelDir);

    FolderDataset fullDataset;
    s = fullDataset.Load(dataFolder);
    if (!s) { lastError_ = s.message; return s; }

    if (fullDataset.Size() == 0) {
        lastError_ = "no training images found in " + dataFolder;
        return Status{StatusCode::ErrorInvalidInput, lastError_};
    }

    std::cout << "Loaded " << fullDataset.Size()
              << " training images (batch)" << std::endl;

    PatchCoreTrainConfig trainCfg;
    trainCfg.inputSize = cfg.inputSize;
    trainCfg.backboneLayers = cfg.backboneLayers;
    trainCfg.backboneType = cfg.backboneType;
    trainCfg.backendType = cfg.backendType;
    trainCfg.coresetFraction = cfg.coresetFraction;
    trainCfg.maxFeatures = cfg.maxFeatures;

    PatchCoreTrainer trainer;

    for (size_t ri = 0; ri < cfg.rois.size(); ri++) {
        const auto& roi = cfg.rois[ri];
        std::cout << "[" << (ri + 1) << "/" << cfg.rois.size()
                  << "] Training ROI: " << roi.id
                  << " (" << roi.w << "x" << roi.h << ")"
                  << std::endl;

        class RoiDataset : public IDataset {
        public:
            Status Load(const std::string&) override { return Status{}; }
            Status AddSample(const Sample& s) {
                samples_.push_back(s);
                return Status{};
            }
            size_t Size() const override { return samples_.size(); }
            Sample Get(size_t index) override { return samples_.at(index); }
            int NumClasses() const override { return 1; }
        private:
            std::vector<Sample> samples_;
        };

        RoiDataset roiDataset;
        cv::Rect roiRect(roi.x, roi.y, roi.w, roi.h);

        for (size_t i = 0; i < fullDataset.Size(); i++) {
            auto fullSample = fullDataset.Get(i);
            cv::Mat roiCrop;
            try {
                roiCrop = fullSample.image(roiRect).clone();
            } catch (const cv::Exception& e) {
                std::cerr << "  Warning: ROI " << roi.id
                          << " out of bounds on image " << i
                          << ": " << e.what() << std::endl;
                continue;
            }

            Sample roiSample;
            roiSample.image = roiCrop;
            roiSample.label = 0;
            roiDataset.AddSample(roiSample);
        }

        if (roiDataset.Size() == 0) {
            std::cerr << "  Warning: ROI " << roi.id
                      << " has no valid samples, skipping" << std::endl;
            continue;
        }

        std::string modelPath = cfg.modelDir + "/" + roi.id + ".bin";
        s = trainer.Train(roiDataset, cfg.backboneModelPath,
                          modelPath, trainCfg);
        if (!s) {
            lastError_ = "ROI " + roi.id + " training failed: " + s.message;
            return s;
        }

        std::cout << "  -> Saved to " << modelPath
                  << " (" << roiDataset.Size() << " samples)"
                  << std::endl;
    }

    return Status{};
}

// -------------------------------------------------------
// 每图 ROI 模式：按图片独立读取 ROI 坐标，按编号分组训练
//
// 工作流程：
//   1. 加载所有图片路径 + 对应的 per-image ROI JSON
//   2. 遍历所有图片，按 ROI id 分组累积特征
//   3. 对每个唯一的 roi id，合并所有图片中该编号 ROI 的特征
//   4. 执行 Coreset 采样 + 构建 MemoryBank + 保存
//
// 输出：每个唯一 id 生成一个 .bin 模型文件
// -------------------------------------------------------
Status RoiTrainer::TrainAllPerImage(const std::string& configPath,
                                     const std::string& dataFolder,
                                     const std::string& perImageRoisDir) {
    namespace fs = std::filesystem;

    // ---- 1. 加载 backbone 配置 ----
    MultiRoiConfig cfg;
    auto s = cfg.FromJson(configPath);
    if (!s) { lastError_ = s.message; return s; }

    fs::create_directories(cfg.modelDir);

    // ---- 2. 扫描训练图片 ----
    std::vector<std::string> imagePaths;
    for (auto& entry : fs::directory_iterator(dataFolder)) {
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" &&
            ext != ".bmp" && ext != ".tiff" && ext != ".tif") {
            continue;
        }
        imagePaths.push_back(entry.path().string());
    }
    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        lastError_ = "no training images found in " + dataFolder;
        return Status{StatusCode::ErrorInvalidInput, lastError_};
    }
    std::cout << "Found " << imagePaths.size() << " training images (per-image ROI mode)"
              << std::endl;

    // ---- 3. 初始化 backbone（共享，只一次） ----
    NodeConfig backboneConfig;
    backboneConfig["model_path"] = cfg.backboneModelPath;
    backboneConfig["backbone_layers"] = cfg.backboneLayers;
    backboneConfig["input_size"] = std::to_string(cfg.inputSize);
    if (cfg.backboneType == "model_backend") {
        backboneConfig["backend_type"] = cfg.backendType;
    }

    auto backbone = CreateBackbone(cfg.backboneType, backboneConfig);
    if (!backbone) {
        lastError_ = "unknown backbone: " + cfg.backboneType;
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
    s = backbone->Init(backboneConfig);
    if (!s) return s;

    // ---- 4. 为每张图读取 ROI 坐标，按编号分组累积特征 ----
    // roiId -> 累积特征列表
    std::unordered_map<std::string, std::vector<PatchFeature>> idFeatures;

    for (size_t fi = 0; fi < imagePaths.size(); fi++) {
        auto img = cv::imread(imagePaths[fi]);
        if (img.empty()) {
            std::cerr << "  Warning: cannot read " << imagePaths[fi]
                      << ", skipping" << std::endl;
            continue;
        }

        std::string basename = fs::path(imagePaths[fi]).filename().string();

        // 查找对应的 per-image ROI JSON
        // 策略：在 perImageRoisDir 下找同名 JSON 文件
        std::string roiJsonPath = perImageRoisDir + "/" +
            fs::path(basename).stem().string() + ".json";

        PerImageRoiConfig picRois;
        s = PerImageRoiConfig::FromJson(roiJsonPath, picRois);
        if (!s) {
            // 如果找不到对应 JSON，跳过这张图
            std::cerr << "  Warning: no ROI config for " << basename
                      << " (" << roiJsonPath << "), skipping" << std::endl;
            continue;
        }

        if ((fi + 1) % 50 == 0) {
            std::cout << "  Processed " << (fi + 1) << "/"
                      << imagePaths.size() << " images" << std::endl;
        }

        // 对这张图的每个 ROI 提取特征，按 id 分组
        for (auto& roi : picRois.rois) {
            cv::Rect roiRect(roi.x, roi.y, roi.w, roi.h);

            // 检查 ROI 是否越界
            if (roiRect.x < 0 || roiRect.y < 0 ||
                roiRect.x + roiRect.width > img.cols ||
                roiRect.y + roiRect.height > img.rows) {
                std::cerr << "  Warning: ROI " << roi.id
                          << " on " << basename << " out of bounds, skipping"
                          << std::endl;
                continue;
            }

            cv::Mat roiCrop = img(roiRect).clone();
            auto feats = backbone->Extract(roiCrop);
            if (!feats.empty()) {
                idFeatures[roi.id].insert(
                    idFeatures[roi.id].end(), feats.begin(), feats.end());
            }
        }
    }

    // ---- 5. 对每个 ROI id 执行 Coreset 采样 + 保存 ----
    PatchCoreTrainConfig trainCfg;
    trainCfg.inputSize = cfg.inputSize;
    trainCfg.backboneLayers = cfg.backboneLayers;
    trainCfg.backboneType = cfg.backboneType;
    trainCfg.backendType = cfg.backendType;
    trainCfg.coresetFraction = cfg.coresetFraction;
    trainCfg.maxFeatures = cfg.maxFeatures;

    // 收集所有唯一的 roi id 并排序（保证输出顺序一致）
    std::vector<std::string> roiIds;
    roiIds.reserve(idFeatures.size());
    for (auto& [id, _] : idFeatures) {
        roiIds.push_back(id);
    }
    std::sort(roiIds.begin(), roiIds.end());

    std::cout << "\n=== Training " << roiIds.size() << " ROI models ===" << std::endl;

    int modelIdx = 0;
    for (auto& roiId : roiIds) {
        modelIdx++;
        auto& feats = idFeatures[roiId];
        std::cout << "[" << modelIdx << "/" << roiIds.size()
                  << "] Training ROI model: " << roiId
                  << " (" << feats.size() << " features from all images)"
                  << std::endl;

        if (feats.empty()) {
            std::cerr << "  Warning: ROI " << roiId
                      << " has no features, skipping" << std::endl;
            continue;
        }

        // 截断到 maxFeatures
        if (feats.size() > trainCfg.maxFeatures) {
            std::shuffle(feats.begin(), feats.end(),
                         std::mt19937(std::random_device()()));
            feats.resize(trainCfg.maxFeatures);
        }

        size_t targetSize = static_cast<size_t>(feats.size() *
                              trainCfg.coresetFraction);
        if (targetSize == 0) targetSize = 1;

        CoresetSampler sampler;
        auto indices = sampler.Sample(feats, targetSize);

        MemoryBank bank;
        std::vector<PatchFeature> coreFeatures;
        for (auto idx : indices) {
            coreFeatures.push_back(feats[idx]);
        }
        bank.Build(coreFeatures);

        std::string modelPath = cfg.modelDir + "/" + roiId + ".bin";
        auto saveStatus = bank.Save(modelPath);
        if (!saveStatus) {
            lastError_ = saveStatus.message;
            return saveStatus;
        }

        std::cout << "  -> Saved to " << modelPath
                  << " (" << coreFeatures.size() << " core features)"
                  << std::endl;
    }

    std::cout << "\n=== Done. " << modelIdx << " ROI models trained. ===" << std::endl;
    return Status{};
}

// -------------------------------------------------------
// 判断大图：检查文件是否存在及文件大小（MB）
// -------------------------------------------------------
bool RoiTrainer::IsLargeImage(const std::string& path, long long thresholdMB) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto sz = fs::file_size(path, ec);
    if (ec) return false;
    return (sz / (1024 * 1024)) >= thresholdMB;
}

} // namespace aicore
