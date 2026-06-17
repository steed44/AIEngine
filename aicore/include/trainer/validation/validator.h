// 验证器，在验证集上评估模型性能
// 支持目标检测指标（mAP、精确率、召回率）的计算
#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <memory>
#include <vector>

namespace aicore {

// 验证结果结构体
// 存储模型在验证集上的各项评估指标
struct ValidationResult {
    float map50 = 0;       // mAP@0.5（IoU 阈值 0.5 时的平均精度）
    float map5095 = 0;     // mAP@0.5:0.95（多个 IoU 阈值的平均精度均值）
    float precision = 0;   // 总体精确率
    float recall = 0;      // 总体召回率
    int totalSamples = 0;  // 参与验证的总样本数
};

// 验证器类
// 对数据集逐样本执行推理，对比标注计算各项评估指标
class Validator {
public:
    Validator();

    // 执行验证流程
    // @param dataset   验证数据集
    // @param inferFn   推理函数：输入图像，输出检测结果列表（NodeResult）
    // @param result    [输出] 计算后的验证指标
    // @return 成功返回 Success
    Status Validate(std::shared_ptr<IDataset> dataset,
                    std::function<std::vector<NodeResult>(const cv::Mat&)> inferFn,
                    ValidationResult& result);

    // 获取最近一次验证的错误信息
    // @return 错误描述字符串
    std::string GetLastError() const;

private:
    std::string lastError_;  // 最近一次错误的描述信息
};

} // namespace aicore
