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

struct InferenceRequest {
    std::string modelName;
    std::vector<cv::Mat> inputs;
    std::function<void(StatusCode, std::vector<cv::Mat>, const std::string&)> callback;
    std::shared_ptr<ModelSlot> slot;
};

struct ModelQueue {
    std::vector<InferenceRequest> pending;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
    std::thread batcherThread;
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