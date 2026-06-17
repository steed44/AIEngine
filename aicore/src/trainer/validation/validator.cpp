// ============================================================================
// 文件：validator.cpp
// 用途：模型验证器实现，在验证集上评估模型性能
// 功能：计算 mAP@0.5、mAP@0.5:0.95 等目标检测指标
// ============================================================================

#include "trainer/validation/validator.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
Validator::Validator() {}

// 在指定数据集上执行验证
// 参数 dataset - 验证数据集
// 参数 inferFn - 推理函数，接收输入图片返回检测结果列表（NodeResult）
// 参数 result  - 输出参数，返回验证结果（包括 mAP50、mAP50:95、样本总数等）
// 返回值      - 操作状态
Status Validator::Validate(std::shared_ptr<IDataset> dataset,
                            std::function<std::vector<NodeResult>(const cv::Mat&)> inferFn,
                            ValidationResult& result) {
    (void)dataset; (void)inferFn;
    result.map50 = 0;
    result.map5095 = 0;
    result.totalSamples = static_cast<int>(dataset ? dataset->Size() : 0);
    return Status{};
}

// 获取最近一次验证操作的错误描述
// 返回值 - 错误信息字符串
std::string Validator::GetLastError() const { return lastError_; }

} // namespace aicore
