# InferenceServer 高吞吐推理引擎设计文档

## 目标

构建独立于现有 Pipeline 的全局推理引擎，支持请求队列、动态批处理、多模型共存、模型热替换和 LRU 显存淘汰，满足生产环境高吞吐需求。

## 约束

- **多个不同模型**可以共存（yolov8n, wideresnet 等），互不干扰
- **单个模型在显存中只存在一个版本**，加载新版本时自动替换旧版本
- 推理过程**不允许出错**，替换期间的请求必须排队等待或由旧版本继续服务
- 推理过程**尽可能不卡顿**，允许微秒级切换停顿

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    InferenceServer                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                ModelRegistry                        │  │
│  │  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │ yolov8n      │  │ wideresnet   │               │  │
│  │  │ v1(active)   │  │ v1(active)   │               │  │
│  │  │ refCount=0   │  │ refCount=2   │               │  │
│  │  └──────────────┘  └──────────────┘               │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ModelQueue   │  │ ModelQueue   │  │ ModelQueue   │  │
│  │ (yolov8n)    │  │ (wideresnet) │  │ (patchcore)  │  │
│  │ ┌────────┐   │  │ ┌────────┐   │  │ ┌────────┐   │  │
│  │ │pending │   │  │ │pending │   │  │ │pending │   │  │
│  │ │timer   │   │  │ │timer   │   │  │ │timer   │   │  │
│  │ │batch   │   │  │ │batch   │   │  │ │batch   │   │  │
│  │ └────────┘   │  │ └────────┘   │  │ └────────┘   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         ▼                 ▼                 ▼          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ModelBackend │  │ ModelBackend │  │ ModelBackend │  │
│  │ (VRAM)      │  │ (VRAM)       │  │ (VRAM)       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
         ▲                                  ▲
         │                                  │
  aicore_server_infer()           回调 → caller
```

## 数据结构

### InferenceRequest

```cpp
struct InferenceRequest {
    std::string modelName;
    std::vector<cv::Mat> inputs;
    std::function<void(StatusCode, std::vector<cv::Mat>, const std::string&)> callback;
    std::shared_ptr<ModelSlot> slot;  // 由 InferAsync 填入，ExecuteBatch 据此取 backend
};
```

### ModelSlot

每个模型在注册表中只有一个槽位，持有唯一后端。

```cpp
struct ModelSlot {
    std::string modelName;
    int version = 0;
    std::unique_ptr<IModelBackend> backend;      // VRAM 中的后端
    std::atomic<int> refCount{0};                 // 正在服务的请求数
    std::atomic<uint64_t> lastUsedTime{0};        // 最近 InferAsync 时间戳
    size_t vramMB = 0;                            // 显存占用估算
    std::mutex swapMutex;                         // 替换期间排他锁
};
```

### ModelQueue

```cpp
struct ModelQueue {
    std::vector<InferenceRequest> pending;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
    std::thread batcherThread;
};
```

## ModelRegistry

```cpp
class ModelRegistry {
public:
    // 获取活跃后端的 Handle（shared_ptr<ModelSlot>，refCount++）
    std::shared_ptr<ModelSlot> GetActive(const std::string& name);

    // 加载新模型替换旧模型
    // 新模型先在 staging 中加载，再原子交换到 active 槽位
    Status Replace(const std::string& name, std::unique_ptr<IModelBackend> newBackend,
                   size_t vramMB, int newVersion);

    // LRU 淘汰：卸载最久未用的模型直到释放 neededMB
    Status EvictLRU(size_t neededMB);

    // 卸载指定模型
    void Unload(const std::string& name);

    // 查询状态（JSON）
    std::string List() const;

private:
    mutable std::shared_mutex rwLock_;
    std::unordered_map<std::string, std::shared_ptr<ModelSlot>> slots_;
};
```

## 模型替换流程

核心目标：旧版本继续服务直到新版本就绪，请求零失败。

```
Replace("yolov8n", newBackend, vramMB=2048, version=2):
  1. 获取 slots_[name] 的 swapMutex（排他锁）
     → 阻止 batcherThread 在此期间获取后端
  2. 将 newBackend 加载到 staging（CPU 内存或临时 CUDA context）
  3. 等待旧后端 refCount 降为 0（正在服务的推理完成）
  4. 原子交换：
     slot->backend = std::move(newBackend)
     slot->version = newVersion
     （旧 backend 被 unique_ptr 析构，释放 VRAM）
  5. 释放 swapMutex → batcherThread 恢复，新请求走新后端

替换期间的新请求行为：
  - GetActive() 等待 swapMutex（微秒级）
  - 拿到锁后读到新后端，正常入队
  - 队列不拒绝、不丢失、不错误
```

## DynamicBatcher

每个模型独立一个 Batcher 线程。

```
新请求到达 InferAsync:
  1. auto slot = registry.GetActive(name)    // refCount++, lastUsedTime=now
  2. 创建 InferenceRequest，slot 字段填入 slot
  3. { lock(queue.mutex) }
     queue.pending.push_back(request)
  4. queue.cv.notify_one()

batcherThread 主循环:
  while (!queue.stop):
    { unique_lock lock(queue.mutex) }
    cv.wait(lock, [&]{ return !pending.empty() || stop; })
    if (stop) break

    if (pending.size() >= maxBatchSize_):
      auto batch = std::move(pending)
      pending.clear()
      lock.unlock()
      ExecuteBatch(batch)
      continue

    cv.wait_for(lock, maxBatchDelayMs_,
                [&]{ return stop || pending.size() >= maxBatchSize_; })
    if (stop) break

    auto batch = std::move(pending)
    pending.clear()
    lock.unlock()
    ExecuteBatch(batch)

ExecuteBatch(batch):
  1. 取 batch[0].slot（所有请求同 slot）
  2. 合并所有帧为 [N, C, H, W] 张量（尺寸不同时 resize 到 slot 的 inputSize）
  3. slot->backend->Infer(batched_input) → batched_output
  4. 按帧索引拆分结果 → 逐个 callback(status, outputs, "")
  5. slot->refCount--（可能触发 EvictLRU）
```

## LRU 淘汰策略

```
EvictLRU(neededMB):
  1. 收集 slots_ 中 refCount==0 的所有 slot
  2. 按 lastUsedTime 升序排列
  3. 逐个 Unload 直到累计释放 >= neededMB
  4. 若还不够 → 返回 Status::ErrorGpuDevice（显存不足）

Unload(name):
  - 设置 queue.stop=true, queue.cv.notify_all()
  - Join batcherThread
  - 处理 queue.pending 中残留的请求：
    - 逐个回调 callback(ErrorServerStop, {}, "model unloading")
  - queue.pending.clear()
  - 删除 slot（析构 backend，释放 VRAM）
  - 从 slots_ 移除

安全保证:
  - refCount > 0 : 不可卸载（正在推理）
  - refCount = 0 : 可安全卸载
```

## InferenceServer

全局单例。

```cpp
class InferenceServer {
public:
    static InferenceServer& Instance();

    Status LoadModel(const std::string& name, const std::string& modelPath,
                     const std::string& backend, size_t vramMB, int version = 1);
    Status ReplaceModel(const std::string& name, const std::string& modelPath,
                        const std::string& backend, size_t vramMB, int version);
    bool IsLoaded(const std::string& name) const;

    Status InferAsync(InferenceRequest req);
    Status InferSync(const std::vector<cv::Mat>& inputs,
                     std::vector<cv::Mat>& outputs,
                     const std::string& modelName);

    void SetBatchConfig(int maxBatchSize, int maxBatchDelayMs);
    void Shutdown();
    std::string ListModels() const;

private:
    ModelRegistry registry_;
    std::unordered_map<std::string, std::unique_ptr<ModelQueue>> queues_;
    int maxBatchSize_ = 32;
    int maxBatchDelayMs_ = 5;
};
```

`InferSync` 通过 promise/future 走同一队列路径：

```cpp
Status InferenceServer::InferSync(const std::vector<cv::Mat>& inputs,
                                   std::vector<cv::Mat>& outputs,
                                   const std::string& modelName) {
    auto promise = std::make_shared<std::promise<std::pair<StatusCode, std::vector<cv::Mat>>>>();
    auto future = promise->get_future();
    InferAsync({
        .modelName = modelName,
        .inputs = inputs,
        .callback = [promise](StatusCode code, auto results, const auto&) {
            promise->set_value({code, std::move(results)});
        }
    });
    auto [code, results] = future.get();
    if (code != StatusCode::OK)
        return Status{code, "infer failed"};
    outputs = std::move(results);
    return Status{};
}
```

## C API

`AICoreResult` 定义见 `include/api/aicore_api.h`（与 `aicore_pipeline_execute` 共用）。

```cpp
AICORE_C_API int aicore_server_load(const char* modelName, const char* modelPath,
                                     const char* backend, int vramMB);
AICORE_C_API int aicore_server_unload(const char* modelName);
AICORE_C_API int aicore_server_infer(const char* modelName,
                                      const unsigned char* data,
                                      int w, int h, int c,
                                      AICoreResult* out, const char** err);
AICORE_C_API int aicore_server_infer_async(const char* modelName,
                                            const unsigned char* data,
                                            int w, int h, int c,
                                            void (*callback)(int status,
                                                const char* resultJson,
                                                const char* errorOut));
AICORE_C_API const char* aicore_server_list();
AICORE_C_API void aicore_server_shutdown();
```

- `aicore_server_infer`: 内部将 raw bytes 转为 cv::Mat，推理后编码为 AICoreResult JSON
- `aicore_server_infer_async`: 回调携带 status + resultJson + errorOut

## 文件结构

```
include/server/
  inference_server.h      ← InferenceServer + InferenceRequest
  model_registry.h         ← ModelRegistry + ModelSlot
  server_api.h             ← C API 声明

src/server/
  inference_server.cpp    ← InferenceServer 实现
  model_registry.cpp      ← ModelRegistry 实现（含 EvictLRU）
  server_api.cpp          ← C API 实现
```

## CMake 集成

```cmake
set(AICORE_SOURCES
    ...
    src/server/inference_server.cpp
    src/server/model_registry.cpp
    src/server/server_api.cpp
)
```

## 测试

- ModelRegistry：Load → GetActive → InferSync → 验证结果正确
- ModelRegistry：Replace → 替换期间请求排队 → 替换后走新后端
- ModelRegistry：EvictLRU → 验证最久未用模型被卸载
- InferenceServer：多模型并发 InferAsync → 验证无竞争
- C API：aicore_server_list → 验证返回 JSON 含所有模型状态

## 实现优先级

| 顺序 | 组件 | 说明 |
|------|------|------|
| 1 | ModelRegistry | 单槽模型管理 + 替换锁 + LRU |
| 2 | DynamicBatcher | 单模型排队 + 超时 batch |
| 3 | InferenceServer | 串联注册表 + 批处理 |
| 4 | C API | 导出接口 |
| 5 | 测试 | 单测 + 集成 |
