// ============================================================
// multi_roi_node.h — 多 ROI PatchCore 推理节点
// 功能：从大图上裁剪多个 ROI 区域，分别用各自模型检测异常，
//       在原始大图上叠加 ROI 框和热力图
// 依赖：共享 IBackbone（特征提取）+ 多份 MemoryBank（模型）
// ============================================================
#pragma once
#include "core/processor.h"
#include "patchcore/memory_bank.h"
#include "patchcore/backbone.h"
#include "patchcore/roi_def.h"
#include <vector>
#include <memory>

namespace aicore {

// -------------------------------------------------------
// RoiModelSlot — 单个 ROI 的推理上下文
// 每个 ROI 拥有独立的 MemoryBank、阈值和裁剪矩形
// -------------------------------------------------------
struct RoiModelSlot {
    std::string roiId;          // ROI 标识
    MemoryBank memoryBank;      // 该 ROI 的预训练记忆库
    float anomalyThreshold;     // 该 ROI 的异常判定阈值（默认 0.5）
    cv::Rect rect;              // 在大图中的裁剪矩形
};

// -------------------------------------------------------
// MultiRoiNode — 多 ROI PatchCore 推理节点
// 职责：作为 IProcessor 节点，接收一帧大图 → 对每个 ROI
//       裁剪 → 提取特征 → 比对 MemoryBank → 叠加结果
//
// 输入：一帧包含完整大图的 Frame
// 输出：Frame 中追加：
//   - roiMap["roi_{id}_score"] = 异常得分
//   - roiMap["roi_{id}_anomaly"] = 是否异常（0/1）
//   - image 上绘制了 ROI 矩形框 + 异常热力图（可选）
// -------------------------------------------------------
class AICORE_API MultiRoiNode : public IProcessor {
public:
    ~MultiRoiNode() override = default;

    // 初始化：加载共享 backbone，加载所有 ROI 的 MemoryBank
    // 配置参数（NodeConfig 格式）：
    //   - config_path: MultiRoiConfig JSON 路径（必需）
    Status Init(const NodeConfig& config) override;

    // 推理一帧大图：遍历所有 ROI，执行裁剪→提取→比对→叠加
    // @param inputs  输入帧列表（取第一帧）
    // @param outputs 输出帧列表（追加结果帧）
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;

    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "multi_roi_patchcore"; }

private:
    // 为单个 ROI 执行推理（裁剪 → 提取 → 比对）
    Status ProcessOneRoi(const cv::Mat& fullImage,
                         const RoiModelSlot& slot,
                         float& outScore, cv::Mat& outHeatmap);

    // 在原图上绘制 ROI 矩形框和异常信息
    // @param image   [in/out] 要绘制的大图
    // @param slot    ROI 信息
    // @param score   异常得分
    void DrawRoiOverlay(cv::Mat& image, const RoiModelSlot& slot, float score);

    std::string name_ = "multi_roi";
    std::unique_ptr<IBackbone> backbone_;      // 共享 backbone（所有 ROI 共用）
    std::vector<RoiModelSlot> slots_;          // 各 ROI 的推理上下文
    MultiRoiConfig config_;                    // 完整配置
    bool drawOverlay_ = true;                  // 是否在原图上绘制检测结果
};

} // namespace aicore
