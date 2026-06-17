// ============================================================
// roi_trainer.h — 多 ROI PatchCore 训练器
// 功能：对一张大图上的多个 ROI 区域分别训练 PatchCore 模型
// 训练流程：读取所有训练大图 → 对每个 ROI 裁剪子图 →
//           → 提取特征 → Coreset 采样 → 构建 MemoryBank → 保存
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
// -------------------------------------------------------
class AICORE_API RoiTrainer {
public:
    RoiTrainer();

    // 训练所有 ROI（默认入口：自动检测大图选择流式或批量模式）
    Status TrainAll(const std::string& configPath,
                    const std::string& dataFolder);

    // 强制流式模式（单次遍历磁盘，每只读一次）
    Status TrainAllStreaming(const std::string& configPath,
                             const std::string& dataFolder);

    // 强制批量模式（一次性全部加载到内存，小图更快）
    Status TrainAllBatch(const std::string& configPath,
                         const std::string& dataFolder);

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
