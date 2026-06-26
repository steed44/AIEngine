// 推理服务器 — 多模型并发推理 + 动态批处理
//
// 架构模式：单例 + 生产者-消费者
//   生产者：InferAsync / InferSync 提交 InferenceRequest 到 ModelQueue
//   消费者：BatcherLoop 线程从队列取出请求，合并为 batch 执行
//
// InferenceRequest：
//   携带模型名、输入图像、回调函数和 ModelSlot（引用计数控制生命周期）。
//   InferSync 通过 Promise/Future 将异步回调转为同步等待。
//
// ModelQueue：
//   每个模型对应一个队列和一个 BatcherLoop 线程（1:1 关系）。
//   队列使用 condition_variable 实现阻塞等待，支持超时批处理。
#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/core.hpp>
#include "core/types.h"
#include "server/model_registry.h"

namespace aicore {

// 推理请求结构体
// 包含模型名、输入图像数据、完成回调函数和 ModelSlot 引用
struct InferenceRequest {
    std::string modelName;                                              // 目标模型名称
    std::vector<cv::Mat> inputs;                                        // 输入图像列表（单帧或多帧）
    std::function<void(StatusCode, std::vector<cv::Mat>, const std::string&)> callback;  // 完成回调
    std::shared_ptr<ModelSlot> slot;                                    // 模型 slot（持有引用计数防止热替换时被销毁）
};

// 模型请求队列
// 每个模型独立拥有一个队列和消费线程，实现模型级隔离
struct ModelQueue {
    std::vector<InferenceRequest> pending;   // 待处理请求缓冲区（批量执行）
    std::mutex mutex;                         // 保护 pending 队列的互斥锁
    std::condition_variable cv;               // 通知 BatcherLoop 线程有新请求
    bool stop = false;                        // 关闭标志（Shutdown 时置 true）
    std::thread batcherThread;                // 该模型的批量调度线程
};

class InferenceServer {
public:
    static InferenceServer& Instance();

    Status LoadModel(const std::string& name, const std::string& modelPath,
                     const std::string& backend, size_t vramMB, int version = 1);
    Status ReplaceModel(const std::string& name, const std::string& modelPath,
                        const std::string& backend, size_t vramMB, int version = 0);
    bool IsLoaded(const std::string& name) const;

    Status InferAsync(InferenceRequest req);
    Status InferSync(const std::vector<cv::Mat>& inputs,
                     std::vector<cv::Mat>& outputs,
                     const std::string& modelName);

    void SetBatchConfig(int maxBatchSize, int maxBatchDelayMs);
    void Shutdown();
    std::string ListModels() const;

private:
    InferenceServer() = default;
    ~InferenceServer();
    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    void BatcherLoop(const std::string& modelName);
    void ExecuteBatch(std::vector<InferenceRequest>& batch);

    ModelRegistry registry_;
    std::unordered_map<std::string, std::unique_ptr<ModelQueue>> queues_;
    int maxBatchSize_ = 32;
    int maxBatchDelayMs_ = 5;
};

} // namespace aicore