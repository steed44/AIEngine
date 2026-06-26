// 推理服务器 — 多模型并发推理 + 动态批处理
//
// 服务器生命周期：
//   1. 启动：单例 Instance() 创建（Meyer's Singleton）
//   2. 加载模型：LoadModel → BackendFactory + ModelRegistry + 启动 BatcherLoop 线程
//   3. 推理请求：InferAsync（异步）或 InferSync（同步，Promise/Future 封装）
//   4. 关闭：Shutdown → 通知所有 BatcherLoop 退出 → 清理残留请求
//
// 动态批处理（Dynamic Batcher）：
//   BatcherLoop 在独立线程中运行，等待条件变量。
//   策略：pending 队列达到 maxBatchSize_ 立即触发，否则等待 maxBatchDelayMs_ 超时。
//   这种"时间+大小"双触发机制平衡了延迟和吞吐量。
//
// 异步执行（Promise/Future 模式）：
//   InferSync 创建 std::promise，将 promise 捕获到 callback lambda 中。
//   InferAsync 将请求入队，BatcherLoop 执行后通过 callback 设置 promise 值。
//   调用方通过 future.get() 阻塞等待结果。
#include "server/inference_server.h"
#include "backend/backend_factory.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <algorithm>

namespace aicore {

// 获取服务器单例（Meyer's Singleton，C++11 起线程安全）
InferenceServer& InferenceServer::Instance() {
    static InferenceServer instance;
    return instance;
}

// 析构自动关闭服务器，等待所有 BatcherLoop 线程退出
InferenceServer::~InferenceServer() {
    Shutdown();
}

// 加载模型：创建后端 → 加载权重 → 注册到 ModelRegistry → 启动 BatcherLoop
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
    // 每个模型有独立的队列和消费线程（1:1 隔离）
    if (queues_.find(name) == queues_.end()) {
        auto q = std::make_unique<ModelQueue>();
        q->stop = false;
        q->batcherThread = std::thread(&InferenceServer::BatcherLoop, this, name);
        queues_[name] = std::move(q);
    }
    return Status{};
}

// 热替换模型：不重启服务器即可切换模型版本
// version ≤ 0 时自动递增，实现无缝升级
Status InferenceServer::ReplaceModel(const std::string& name, const std::string& modelPath,
                                      const std::string& backend, size_t vramMB, int version) {
    // version <= 0 时自动递增：从注册表取当前 version + 1，不存在则从 1 开始
    if (version <= 0) {
        version = registry_.Contains(name) ? registry_.GetVersion(name) + 1 : 1;
    }
    fprintf(stderr, "[InferenceServer] ReplaceModel: %s -> %s (v%d)\n",
            name.c_str(), modelPath.c_str(), version);
    return LoadModel(name, modelPath, backend, vramMB, version);
}

bool InferenceServer::IsLoaded(const std::string& name) const {
    return registry_.Contains(name);
}

// 异步推理：请求入队 → BatcherLoop 线程取出 → 回调返回结果
// 请求通过 ModelRegistry 获取 slot（引用计数 +1），防止热替换时 backend 被销毁
Status InferenceServer::InferAsync(InferenceRequest req) {
    // 注册表获取活跃 slot（refCount++）
    // 持有 slot 引用确保推理期间 backend 不会被 EvictLRU 淘汰
    auto slot = registry_.GetActive(req.modelName);
    if (!slot) {
        return Status{StatusCode::ErrorModelLoad, "model not loaded: " + req.modelName};
    }
    req.slot = slot;

    auto it = queues_.find(req.modelName);
    if (it == queues_.end()) {
        return Status{StatusCode::ErrorInternal, "no queue for: " + req.modelName};
    }
    // 请求入队后通知 BatcherLoop 线程取任务
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

// 同步推理：Promise/Future 模式将异步调用封装为同步
// 创建 promise → 捕获到 callback → InferAsync → future.get() 阻塞等待
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

// BatcherLoop 线程：等待条件变量 → 积累 batch → ExecuteBatch
// "时间+大小"双触发：
//   大小触发：pending >= maxBatchSize_ 立即执行
//   时间触发：未达到大小但 maxBatchDelayMs_ 超时也执行
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
                    [this, &q] { return q->stop || q->pending.size() >= (size_t)maxBatchSize_; });
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

// 执行批量推理：遍历 batch 中的每个请求，逐帧调用后端 Infer
// 当前实现是"伪批处理"（逐帧推理），非真正的动态 batch
// 未来优化：合并多帧为一个大 tensor，后端一次推理返回 batch 结果
void InferenceServer::ExecuteBatch(std::vector<InferenceRequest>& batch) {
    if (batch.empty()) return;
    auto& slot = batch[0].slot;
    if (!slot || !slot->backend) {
        for (auto& req : batch) {
            req.callback(StatusCode::ErrorModelLoad, {}, "backend unavailable");
            registry_.Release(req.slot);
        }
        return;
    }

    // 逐帧推理（当前为简化实现）
    // 每帧经历：resize → float 归一化 → HWC→CHW 转置 → Infer → tensor→Mat
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
            // HWC → CHW 转换：用 cv::split + cv::merge 替代三重循环（SIMD 加速）
            std::vector<cv::Mat> channels(3);
            cv::split(resized, channels);
            std::vector<float> chw(3 * 224 * 224);
            for (int c = 0; c < 3; c++) {
                // 每个通道是 CV_32FC1 连续内存，直接 memcpy
                std::memcpy(chw.data() + c * 224 * 224,
                            channels[c].ptr<float>(), 224 * 224 * sizeof(float));
            }
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
        registry_.Release(req.slot);
    }
}

void InferenceServer::SetBatchConfig(int maxBatchSize, int maxBatchDelayMs) {
    maxBatchSize_ = maxBatchSize;
    maxBatchDelayMs_ = maxBatchDelayMs;
}

// 关闭服务器：停止所有 BatcherLoop → 等待线程退出 → 清理残留请求 → 释放队列
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
            if (req.slot) registry_.Release(req.slot);
        }
        q->pending.clear();
    }
    queues_.clear();
}

// 列出所有已加载模型（委托 ModelRegistry::List，返回 JSON 格式）
std::string InferenceServer::ListModels() const {
    return registry_.List();
}

} // namespace aicore