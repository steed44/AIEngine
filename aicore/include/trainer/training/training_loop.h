// 训练主循环，协调数据集、增强、模型和回调的执行
// 负责完整的训练-验证迭代流程控制
#pragma once
#include "core/types.h"
#include "trainer/data/data_loader.h"
#include "trainer/data/augmentation.h"
#include "trainer/model/model_factory.h"
#include "trainer/callback.h"
#include <memory>
#include <vector>

namespace aicore {

// 训练配置结构体
// 包含训练过程的所有可调超参数和环境设置
struct TrainConfig {
    int epochs = 100;                  // 总训练轮数
    float learningRate = 0.001f;       // 初始学习率
    float weightDecay = 0.0001f;       // 权重衰减系数（L2 正则化）
    int batchSize = 16;                // 每批样本数量
    int numClasses = 10;               // 分类任务类别数
    int inputSize = 224;               // 输入图像尺寸（宽高相同）
    std::string saveDir = "checkpoints";  // 模型检查点保存目录
    std::vector<int> gpuIds = {0};     // 使用的 GPU 设备 ID 列表
};

// 训练循环控制器
// 管理完整的训练流程：数据加载、前向传播、反向传播、验证、回调通知
class TrainingLoop {
public:
    TrainingLoop();

    // 启动训练循环
    // @param config         训练超参数配置
    // @param trainDataset   训练数据集
    // @param valDataset     验证数据集（用于评估，可为空）
    // @param trainAug       训练时使用的数据增强流水线
    // @return 成功返回 Success
    Status Run(const TrainConfig& config,
               std::shared_ptr<IDataset> trainDataset,
               std::shared_ptr<IDataset> valDataset,
               AugmentationPipeline& trainAug);

    // 注册训练回调
    // @param callback  回调实例（在训练各阶段自动触发）
    void AddCallback(std::shared_ptr<ITrainCallback> callback);

    // 请求停止训练（安全中断正在运行的训练）
    // @return 成功返回 Success
    Status Stop();

private:
    std::vector<std::shared_ptr<ITrainCallback>> callbacks_;  // 已注册的回调列表
    bool running_ = false;   // 训练是否正在运行
};

} // namespace aicore
