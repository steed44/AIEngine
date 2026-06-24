#include "server/inference_server.h"
#include "backend/backend_factory.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <algorithm>

namespace aicore {

InferenceServer& InferenceServer::Instance() {
    static InferenceServer instance;
    return instance;
}

InferenceServer::~InferenceServer() {
    Shutdown();
}

Status InferenceServer::LoadModel(const std::string& name, const std::string& modelPath,
                                   const std::string& backend, size_t vramMB, int version) {
    BackendType bt = BackendType::kONNXRuntime;
    if (backend == "tensorrt") bt = BackendType::kTensorRT;
    else if (backend == "libtorch") bt = BackendType::kLibTorch;

    auto modelBackend = BackendFactory::Create(bt);
    if (!modelBackend) {
        return Status{StatusCode::ErrorModelLoad, "unknown backend: " + backend};
    }

    ModelInfo info;
    info.modelPath = modelPath;
    auto s = modelBackend->Load(info);
    if (!s) return s;

    s = registry_.Replace(name, std::move(modelBackend), vramMB, version);
    if (!s) return s;

    // 创建模型队列和 batcher 线程
    if (queues_.find(name) == queues_.end()) {
        auto q = std::make_unique<ModelQueue>();
        q->stop = false;
        q->batcherThread = std::thread(&InferenceServer::BatcherLoop, this, name);
        queues_[name] = std::move(q);
    }
    return Status{};
}

Status InferenceServer::ReplaceModel(const std::string& name, const std::string& modelPath,
                                      const std::string& backend, size_t vramMB, int version) {
    return LoadModel(name, modelPath, backend, vramMB, version);
}

bool InferenceServer::IsLoaded(const std::string& name) const {
    // 简单的检查：registry_ 中有此条目即认为已加载
    auto json = registry_.List();
    return json.find("\"" + name + "\"") != std::string::npos;
}

Status InferenceServer::InferAsync(InferenceRequest req) {
    // 注册表获取活跃 slot（refCount++）
    auto slot = registry_.GetActive(req.modelName);
    if (!slot) {
        return Status{StatusCode::ErrorModelLoad, "model not loaded: " + req.modelName};
    }
    req.slot = slot;

    auto it = queues_.find(req.modelName);
    if (it == queues_.end()) {
        return Status{StatusCode::ErrorInternal, "no queue for: " + req.modelName};
    }
    auto& q = it->second;
    {
        std::lock_guard lock(q->mutex);
        if (q->stop) {
            return Status{StatusCode::ErrorResourceExhaust, "queue stopped"};
        }
        q->pending.push_back(std::move(req));
    }
    q->cv.notify_one();
    return Status{};
}

Status InferenceServer::InferSync(const std::vector<cv::Mat>& inputs,
                                   std::vector<cv::Mat>& outputs,
                                   const std::string& modelName) {
    auto promise = std::make_shared<std::promise<std::pair<StatusCode, std::vector<cv::Mat>>>>();
    auto future = promise->get_future();

    InferenceRequest req;
    req.modelName = modelName;
    req.inputs = inputs;
    req.callback = [promise](StatusCode code, std::vector<cv::Mat> results, const std::string&) {
        promise->set_value({code, std::move(results)});
    };

    auto s = InferAsync(std::move(req));
    if (!s) {
        return s;
    }

    auto [code, results] = future.get();
    if (code != StatusCode::OK) {
        return Status{code, "infer failed"};
    }
    outputs = std::move(results);
    return Status{};
}

void InferenceServer::BatcherLoop(const std::string& modelName) {
    auto it = queues_.find(modelName);
    if (it == queues_.end()) return;
    auto& q = it->second;

    while (true) {
        std::vector<InferenceRequest> batch;
        {
            std::unique_lock lock(q->mutex);
            q->cv.wait(lock, [&q] { return !q->pending.empty() || q->stop; });
            if (q->stop && q->pending.empty()) break;

            // 累积批量：立即取所有待处理或等到超时
            if (q->pending.size() >= (size_t)maxBatchSize_) {
                batch = std::move(q->pending);
                q->pending.clear();
            } else {
                q->cv.wait_for(lock, std::chrono::milliseconds(maxBatchDelayMs_),
                    [&q] { return q->stop || q->pending.size() >= (size_t)q->pending.capacity(); });
                batch = std::move(q->pending);
                q->pending.clear();
            }
        }
        if (!batch.empty()) {
            ExecuteBatch(batch);
        }
        if (q->stop) break;
    }
}

void InferenceServer::ExecuteBatch(std::vector<InferenceRequest>& batch) {
    if (batch.empty()) return;
    auto& slot = batch[0].slot;
    if (!slot || !slot->backend) {
        for (auto& req : batch) {
            req.callback(StatusCode::ErrorModelLoad, {}, "backend unavailable");
            slot->refCount.fetch_sub(1);
        }
        return;
    }

    // 合并所有输入为 batch 张量
    // 由于 IModelBackend 接口的 Infer 接受 vector<Tensor>，
    // 逐帧推理后收集结果（简单实现，不做动态 batch）
    for (auto& req : batch) {
        std::vector<cv::Mat> results;
        bool ok = true;
        for (auto& img : req.inputs) {
            // HWC → CHW 转换
            cv::Mat floatImg, resized;
            cv::resize(img, resized, cv::Size(224, 224));
            resized.convertTo(floatImg, CV_32F, 1.0 / 255);

            Tensor input;
            input.dtype = DataType::kFloat32;
            input.shape = {1, 3, 224, 224};
            input.bytes = 1 * 3 * 224 * 224 * sizeof(float);
            std::vector<float> chw(3 * 224 * 224);
            float* src = floatImg.ptr<float>();
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 224; h++)
                    for (int w = 0; w < 224; w++)
                        chw[c * 224 * 224 + h * 224 + w] = src[h * 224 * 3 + w * 3 + c];
            input.data = chw.data();

            std::vector<Tensor> outputs;
            auto s = slot->backend->Infer({input}, outputs);
            if (!s) {
                ok = false;
                req.callback(s.code, {}, s.message);
                break;
            }
            // 将输出 tensor 转回 cv::Mat
            for (auto& t : outputs) {
                if (t.shape.size() >= 2) {
                    int h = (int)t.shape[t.shape.size() - 2];
                    int w = (int)t.shape[t.shape.size() - 1];
                    cv::Mat tmpMat(h, w, CV_32F, t.data);
                    cv::Mat out = tmpMat.clone();
                    results.push_back(out);
                }
            }
        }
        if (ok) {
            req.callback(StatusCode::OK, results, "");
        }
        slot->refCount.fetch_sub(1);
    }
}

void InferenceServer::SetBatchConfig(int maxBatchSize, int maxBatchDelayMs) {
    maxBatchSize_ = maxBatchSize;
    maxBatchDelayMs_ = maxBatchDelayMs;
}

void InferenceServer::Shutdown() {
    for (auto& [name, q] : queues_) {
        {
            std::lock_guard lock(q->mutex);
            q->stop = true;
        }
        q->cv.notify_all();
        if (q->batcherThread.joinable()) {
            q->batcherThread.join();
        }
        // 处理残留请求
        std::lock_guard lock(q->mutex);
        for (auto& req : q->pending) {
            req.callback(StatusCode::ErrorResourceExhaust, {}, "server shutdown");
            if (req.slot) req.slot->refCount.fetch_sub(1);
        }
        q->pending.clear();
    }
    queues_.clear();
}

std::string InferenceServer::ListModels() const {
    return registry_.List();
}

} // namespace aicore