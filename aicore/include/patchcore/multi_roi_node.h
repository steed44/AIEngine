// ============================================================
// multi_roi_node.h — 多 ROI PatchCore 推理节点
// 功能：从大图上裁剪多个 ROI 区域，分别用各自模型检测异常，
//       在原始大图上叠加 ROI 框和热力图
// 依赖：共享 IBackbone（特征提取）+ 多份 MemoryBank（模型）
//
// 两种模式：
//   1. 固定 ROI 模式：从配置加载固定 ROI 坐标
//   2. 每图 ROI 模式：推理时动态加载 per-image ROI JSON
// ============================================================
#pragma once
#include "core/processor.h"
#include "patchcore/tiered_memory_bank.h"
#include "patchcore/backbone.h"
#include "patchcore/roi_def.h"
#include "engine/thread_pool.h"
#include <vector>
#include <memory>

namespace aicore {

// -------------------------------------------------------
// RoiModelSlot — 单个 ROI 的推理上下文
// 每个 ROI 拥有独立的 MemoryBank、阈值和裁剪矩形
// -------------------------------------------------------
struct RoiModelSlot {
    std::string roiId;          // ROI 标识
    TieredMemoryBank memoryBank;      // 该 ROI 的三级存储记忆库
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
//
// 两种初始化模式：
//   1. 固定模式：config["config_path"] 指向固定坐标 JSON
//   2. 每图模式：config["config_path"] + config["per_image_rois_dir"]
//      推理时动态加载 per-image ROI JSON
// -------------------------------------------------------
class AICORE_API MultiRoiNode : public IProcessor {
public:
    ~MultiRoiNode() override = default;

    // 初始化：加载共享 backbone，加载所有 ROI 的 MemoryBank
    // 配置参数（NodeConfig 格式）：
    //   - config_path: MultiRoiConfig JSON 路径（必需）
    //   - per_image_rois_dir: 每图 ROI JSON 目录（每图模式必需）
    //   - draw_overlay: 是否在原图绘制检测结果（可选，"true"/"false"）
    // 前置条件：config 包含 "config_path" 键，指向有效 JSON
    // 后置条件：backbone_ 和 slots_ 初始化完毕，Process() 就绪
    // 线程安全：非线程安全，应在 pipeline 构建阶段单线程调用
    Status Init(const NodeConfig& config) override;

    // 推理一帧大图：遍历所有 ROI，执行裁剪→提取→比对→叠加
    // @param inputs  输入帧列表（取 inputs[0] 作为大图）
    // @param outputs 输出帧列表（追加结果帧，包含 ROI 叠加图）
    // 前置条件：Init() 已成功调用，inputs 非空且第一帧 image 非空
    // 后置条件：outputs 末尾追加包含 ROI 绘制结果的 Frame
    // 线程安全：支持 ThreadPool 并行处理多个 ROI 的 ProcessOneRoi 调用
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;

    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "multi_roi_patchcore"; }

    void SetThreadPool(ThreadPool* pool) override { threadPool_ = pool; }

    // 动态加载 per-image ROI 坐标（每图模式用）
    // @param imageFilename 图片文件名（不含路径）
    // @param roisDir       ROI JSON 目录
    // @return 成功返回 Status{}，失败返回错误
    Status LoadPerImageRois(const std::string& imageFilename,
                            const std::string& roisDir);

private:
    // 为单个 ROI 执行推理（裁剪 → 提取 → 比对）
    // @param fullImage  原始大图
    // @param slot       ROI 推理上下文（包含裁剪矩形和 MemoryBank）
    // @param outScore   [out] 该 ROI 的异常得分
    // @param outHeatmap [out] 该 ROI 的异常热力图（对应裁剪区域尺寸）
    // 前置条件：fullImage 非空，slot 已初始化
    // 后置条件：outScore 填充异常得分，outHeatmap 可选
    Status ProcessOneRoi(const cv::Mat& fullImage,
                         const RoiModelSlot& slot,
                         float& outScore, cv::Mat& outHeatmap);

    // 在原图上绘制 ROI 矩形框和异常信息
    // @param image   [in/out] 要绘制的大图（直接修改此图像）
    // @param slot    ROI 信息（包含裁剪矩形）
    // @param score   异常得分（显示在 ROI 框旁）


    std::string name_ = "multi_roi";
    std::unique_ptr<IBackbone> backbone_;      // 共享 backbone（所有 ROI 共用，仅加载一次）
    std::vector<RoiModelSlot> slots_;          // 各 ROI 的推理上下文（含独立 MemoryBank 和阈值）
    MultiRoiConfig config_;                    // 完整配置（从 JSON 解析）
    bool drawOverlay_ = true;                  // 是否在原图上绘制检测结果（ROI 框 + 异常分数）
    ThreadPool* threadPool_ = nullptr;         // 线程池引用（用于并行处理多个 ROI 的推理）
};

} // namespace aicore
