// ============================================================
// roi_trainer.h — 多 ROI PatchCore 训练器
// 功能：对一张大图上的多个 ROI 区域分别训练 PatchCore 模型
// 训练流程：读取所有训练大图 → 对每个 ROI 裁剪子图 →
//           → 提取特征 → Coreset 采样 → 构建 MemoryBank → 保存
//
// 支持三种训练模式：
//   1. TrainAll()       — 自动选择流式/批量（固定 ROI 坐标）
//   2. TrainAllPerImage() — 每图独立 ROI 坐标，按编号分组训练
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/roi_def.h"
#include <string>

namespace aicore {

// -------------------------------------------------------
// RoiTrainer — 多 ROI 训练器
// 职责：遍历 MultiRoiConfig 中定义的每个 ROI，从训练大图集中
//       裁剪 ROI 区域，分别训练独立的 MemoryBank 模型
//
// 典型使用场景：
//   RoiTrainer trainer;
//   trainer.TrainAll("rois.json", "./normal_pcbs/");
//   // 输出: ./models/connector.bin, ./models/chip.bin, ...
//
//   // 或每图 ROI 模式：
//   trainer.TrainAllPerImage("config.json", "./images/", "./roi_annotations/");
//   // 输出: ./models/1.bin, ./models/2.bin, ..., ./models/10.bin
// -------------------------------------------------------
class AICORE_API RoiTrainer {
public:
    RoiTrainer();

    // 训练所有 ROI（默认入口：自动检测大图选择流式或批量）
    // 自动检测逻辑：遍历 dataFolder，若任何图片 > 100MB 则自动切换流式模式
    // @param configPath  MultiRoiConfig JSON 路径
    // @param dataFolder  训练大图文件夹
    // @return 所有 ROI 训练完成后返回 Status{}
    Status TrainAll(const std::string& configPath,
                    const std::string& dataFolder);

    // 强制流式模式（单次遍历磁盘，每张图只读一次）
    // 适用于大图（>100MB）场景，峰值内存约 1.2GB
    // 训练流程：遍历大图 → 对每个 ROI 裁剪 → Extract → 暂存特征 →
    //   所有图处理完后 → Coreset 采样 → 构建 MemoryBank → 保存
    Status TrainAllStreaming(const std::string& configPath,
                             const std::string& dataFolder);

    // 强制批量模式（一次性全部加载到内存，小图更快）
    // 适用于小图（<100MB）场景，需要足够内存容纳所有图片
    Status TrainAllBatch(const std::string& configPath,
                         const std::string& dataFolder);

    // 每图 ROI 模式：每张图有独立 ROI 坐标 JSON
    // @param configPath     backbone 配置文件路径
    // @param dataFolder     训练大图文件夹
    // @param perImageRoisDir 每图 ROI 坐标 JSON 所在目录
    //   匹配规则：图片 a.png → 查找 a.json
    //   相同 id 的 ROI 跨图合并训练，最终输出 N 个模型文件
    Status TrainAllPerImage(const std::string& configPath,
                            const std::string& dataFolder,
                            const std::string& perImageRoisDir);

    // 强制/禁止流式模式（覆盖自动检测）
    void SetForceStream(bool force) { forceStream_ = force; }
    void SetForceNoStream(bool force) { forceNoStream_ = force; }

    // 获取最后出错信息
    std::string GetLastError() const { return lastError_; }

private:
    static bool IsLargeImage(const std::string& path, long long thresholdMB = 100);

    std::string lastError_;
    bool forceStream_ = false;
    bool forceNoStream_ = false;
};

} // namespace aicore
