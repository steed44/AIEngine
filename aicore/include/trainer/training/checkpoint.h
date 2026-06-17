// 检查点管理器，负责模型权重的保存、加载与清理
// 支持按 epoch 保存、获取最佳模型、自动清理旧检查点
#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

// 检查点管理器类
// 在训练过程中周期性地保存模型状态，支持断点续训和最佳模型提取
class CheckpointManager {
public:
    // 构造检查点管理器
    // @param saveDir  检查点文件的保存目录
    CheckpointManager(const std::string& saveDir);

    // 保存当前 epoch 的检查点
    // @param epoch      当前 epoch 编号
    // @param metric     当前评估指标值（用于判断最佳模型）
    // @param modelData  模型权重数据的序列化字符串
    // @return 成功返回 Success
    Status Save(int epoch, float metric, const std::string& modelData);

    // 加载指定 epoch 的检查点
    // @param epoch      目标 epoch 编号
    // @param modelData  [输出] 加载到的模型权重数据
    // @return 成功返回 Success
    Status Load(int epoch, std::string& modelData);

    // 获取最佳模型（指标最高的检查点）
    // @param modelData  [输出] 最佳模型的权重数据
    // @return 成功返回 Success
    Status GetBest(std::string& modelData);

    // 清理旧检查点，只保留性能最好的 N 个
    // @param keepBest  保留的最佳检查点数量（默认 3）
    // @return 成功返回 Success
    Status Cleanup(int keepBest = 3);

private:
    std::string saveDir_;  // 检查点文件的保存根目录
};

} // namespace aicore
