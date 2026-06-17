// ============================================================
// patchcore_node.h — PatchCore 异常检测算法的处理节点
// 功能：作为管线中的一个处理器节点，接收图像帧并输出异常检测结果
// 依赖：IBackbone（特征提取）+ MemoryBank（记忆库比对）
// ============================================================
#pragma once
#include "core/processor.h"
#include "patchcore/memory_bank.h"
#include "patchcore/backbone.h"
#include <string>
#include <vector>

namespace aicore {

// -------------------------------------------------------
// PatchCoreNode — PatchCore 算法推理节点
// 职责：将输入图像通过 backbone 提取局部特征，与 MemoryBank
//       中存储的正常特征库做最近邻比对，输出异常热力图和评分
// 典型使用场景：作为视频分析管线中的一个 IProcessor 节点，
//       接收 Frame 流，实时输出异常检测结果
// -------------------------------------------------------
class PatchCoreNode : public IProcessor {
public:
    // 初始化节点：加载 backbone 配置、MemoryBank 文件路径、
    //             输入尺寸和异常判定阈值
    Status Init(const NodeConfig& config) override;
    // 处理一帧图像：提取特征 → 比对记忆库 → 生成异常热力图
    // @param inputs  输入帧列表（取第一帧的图像）
    // @param outputs 输出帧列表（追加一帧，含 anomaly_score / is_anomaly）
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "patchcore"; }

private:
    std::string name_;                              // 节点名称
    std::unique_ptr<IBackbone> backbone_;            // 特征提取 backbone
    MemoryBank memoryBank_;                          // 正常样本特征记忆库
    int inputSize_ = 224;                            // 输入图像缩放尺寸
    float anomalyThreshold_ = 0.5f;                  // 异常判定阈值
};

} // namespace aicore
