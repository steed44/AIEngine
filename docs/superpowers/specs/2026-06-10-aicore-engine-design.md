# AI Core 推理引擎设计文档

## 概述

基于 NVIDIA GPU 工控机的工业检测纯 C++ AI 引擎，采用 Pipeline + Strategy 设计模式，支持多模型串并联 DAG 编排，配置驱动，面向 Windows (VS2022) + Qt 5.12.11 + OpenCV 4.7.0 开发环境。

## 总体架构

```
Qt 上位机应用层
    │ C 接口 (DLL, Create/Destroy 配对)
    ▼
推理引擎核心 (aicore.dll)
    ├── Parser 配置解析层 (JSON → PipelineConfig)
    ├── Pipeline 编排执行层 (DAG 调度 / 串并联 / 拓扑排序)
    ├── Backend 适配层 (Strategy: TensorRT / ONNX Runtime / LibTorch)
    └── 预处理 / 后处理节点工厂
```

## 公共类型定义

```cpp
// DLL 导出宏
#ifdef AICORE_EXPORTS
#define AICORE_API __declspec(dllexport)
#else
#define AICORE_API __declspec(dllimport)
#endif

// 数据布局
enum class MemoryType { kCPU, kGPU, kPinned };

// 精度类型
enum class DataType { kUInt8, kFloat32, kFloat16 };

// 张量（采用显存分配器统一管理生命周期）
struct Tensor {
    DataType dtype;
    std::vector<int64_t> shape;
    MemoryType memory;
    void* data;       // 指向分配器管理的显存/内存块，不拥有所有权
    size_t bytes;
    size_t allocId;   // 分配器中的块 ID，用于统一释放
};

// 配置类型（明确使用 nlohmann::json）
using Config = nlohmann::json;

// 状态码
enum class StatusCode {
    OK = 0,
    Skip,               // 上游失败，本节点跳过
    ErrorConfigParse,
    ErrorModelLoad,
    ErrorModelInfer,
    ErrorPreprocess,
    ErrorPostprocess,
    ErrorResourceExhaust,
    ErrorTimeout,
    ErrorInvalidInput,
    ErrorInternal,
    ErrorGpuDevice
};

struct Status {
    StatusCode code;
    std::string message;
    operator bool() const { return code == StatusCode::OK; }
};
```

## 核心接口

### IProcessor — 所有处理节点统一接口

```cpp
class IProcessor {
public:
    virtual ~IProcessor() = default;
    virtual std::string name() const = 0;
    virtual Status init(const Config& config) = 0;
    virtual Status process(const Frame& input, Frame& output) = 0;
    virtual Status destroy() = 0;
};
```

### Frame — 核心数据传递单元

```cpp
// 流水线中节点间传递的数据，零拷贝设计
struct Frame {
    uint64_t id;                    // 帧序号
    cv::Mat image;                  // 原始图像（CPU）
    int width;                      // 原始宽度
    int height;                     // 原始高度
    int channels;                   // 原始通道数
    std::vector<Tensor> gpuTensors; // GPU 侧张量
    std::map<std::string, NodeResult> nodeResults; // 各节点结果
    std::map<std::string, Status> nodeStatuses;    // 各节点执行状态
};
```

### IModelBackend — 推理后端统一抽象

```cpp
// IModelBackend 继承 IProcessor，但 process() 不是给外部直接调用的
// 约定：IModelBackend 的 process() 实现为 return ErrorInternal（标记为不应直接调用）
//       ModelNode 在 process() 中编排：预处理 → IModelBackend::infer() → 后处理
//       infer() 仅负责 GPU 推理计算
class IModelBackend : public IProcessor {
public:
    virtual ModelInfo modelInfo() const = 0;
    virtual Status infer(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) = 0;
};

struct ModelInfo {
    std::string name;
    std::vector<IOInfo> inputs;   // 输入张量名、形状、数据类型
    std::vector<IOInfo> outputs;  // 输出张量名、形状、数据类型
    BackendType backend;
    std::string modelPath;
};

enum class BackendType { kTensorRT, kONNXRuntime, kLibTorch };
```

### PipelineConfig — 管道配置

```cpp
// Parser 层的产出物，由 ConfigParser 从 JSON 解析得到
struct NodeConfig {
    std::string id;
    std::string type;       // model / preprocess / postprocess / merge / composite
    std::string name;
    std::string backend;    // 仅 type=model
    std::string modelPath;  // 仅 type=model
    int inputWidth = 0;
    int inputHeight = 0;
    std::vector<Config> preprocessSteps;  // 预处理流水线描述
    std::vector<Config> postprocessSteps; // 后处理流水线描述
    Config params;           // 自定义参数
    bool required = false;
    size_t enginePoolSize = 3;
};

struct EdgeConfig {
    std::string from;
    std::string to;
};

struct PipelineConfig {
    std::string name;
    std::vector<NodeConfig> nodes;
    std::vector<EdgeConfig> edges;
};
```

### IPipeline — 流水线编排器

```cpp
class IPipeline {
public:
    virtual ~IPipeline() = default;
    virtual Status build(const PipelineConfig& config) = 0;
    // 拓扑排序在 build() 时缓存，run() 时直接使用
    virtual Status run(Frame& frame) = 0;
};
```

## Pipeline 编排层

### 节点类型

| 节点类型 | 说明 |
|---------|------|
| ModelNode | 推理节点，内聚预处理→IModelBackend::infer→后处理 |
| PreprocessNode | 预处理节点（resize / normalize / cvtColor 等） |
| PostprocessNode | 后处理节点（nms / softmax / label_map 等） |
| CompositeNode | 组合多个子节点，支持 Serial 和 Parallel 模式。Serial 模式是子流水线，内部按序执行多步操作 |
| MergeNode | 合并多个上游节点输出。策略包括：Union（合并列表）、MaxScore（取最高置信度）、Average（平均） |
| Pipeline | 顶层容器，管理完整 DAG。build() 时做拓扑排序并缓存执行顺序 |

### CompositeNode 与 Pipeline 的关系

- Pipeline 是顶级编排器，管理完整 DAG，包含所有节点和边
- CompositeNode 是局部子组合，用于封装一组有逻辑关系的串/并联节点
- CompositeNode 可嵌套在 Pipeline 中，也可嵌套在其他 CompositeNode 中
- Pipeline 不可嵌套

### DAG 执行策略

1. `build()` 时：建立邻接表，拓扑排序，缓存执行顺序链表
2. 按拓扑序分层：同一层无依赖关系的节点可并行执行
3. 每层内部用线程池调度多个节点并行
4. 层间串行等待
5. 节点执行结果写入 Frame.nodeResults / Frame.nodeStatuses

### 失败传播

- 节点返回 Error 状态码 → 写入 Frame.nodeStatuses
- 下游节点检测到依赖节点状态为 Error → 自身设为 Skip，不执行 process()
- 节点配置 `"required": true` → 该节点失败则整条流水线终止，返回 Error
- 节点配置 `"required": false`（默认） → 下游节点自动 Skip，继续执行

### 拓扑排序缓存

- `build()` 时完成拓扑排序，按序存入 `std::vector<size_t> execOrder_`
- `run()` 直接遍历 `execOrder_` 执行，O(n) 无重复计算

### DAG 校核

- 环检测：DFS 检查是否存在反向边
- 连通性：所有节点至少有一条路径连接到 input 或 output
- IO 类型匹配（运行时校验）：边两端节点的输出/输入数据类型是否兼容

## 配置驱动

### JSON 配置 Schema

```json
{
  "pipeline": {
    "name": "string",
    "nodes": [
      {
        "id": "string",              // 节点唯一标识
        "type": "string",            // 节点类型: model / preprocess / postprocess / merge / composite
        "name": "string",            // 业务名称
        "backend": "string",         // 仅 type=model: tensorrt / onnxruntime / libtorch
        "model": "string",           // 仅 type=model: 模型文件路径
        "input": {                   // 仅 type=model: 模型输入配置
          "width": 640,
          "height": 640
        },
        "preprocess": [              // 仅 type=model: 预处理流水线
          {"type": "resize", "width": 640, "height": 640},
          {"type": "normalize", "mean": [0,0,0], "std": [1,1,1]}
        ],
        "postprocess": [             // 仅 type=model: 后处理流水线
          {"type": "nms", "iou_threshold": 0.5, "conf_threshold": 0.3}
        ],
        "params": {},                // 各节点自定义参数
        "required": false,           // 是否关键节点
        "engine_pool_size": 3        // 仅 type=model: 引擎池大小
      }
    ],
    "edges": [
      {"from": "nodeA", "to": "nodeB"}           // 普通边
    ]
  }
}
```

### PipelineBuilder

```cpp
class PipelineBuilder {
    // 工厂注册
    using CreatorFunc = std::function<std::unique_ptr<IProcessor>(const Config&)>;
    std::unordered_map<std::string, CreatorFunc> registry_;

public:
    void registerNodeType(const std::string& type, CreatorFunc creator);

    // build() 返回 unique_ptr 明确所有权
    std::unique_ptr<Pipeline> build(const PipelineConfig& config);
};
```

### 插件注册机制

```cpp
// 可选的静态注册宏（仅用于编译期注入，不跨 DLL 边界）
#define REGISTER_PROCESSOR(type, creatorClass) \
    namespace { \
    struct StaticRegistrar_##creatorClass { \
        StaticRegistrar_##creatorClass() { \
            /* 此处仅做简单注册，不加锁；全局静态构造串行执行 */ \
            PipelineBuilder::globalInstance().registerNodeType(type, \
                [](const Config& cfg) { return std::make_unique<creatorClass>(cfg); }); \
        } \
    }; \
    static StaticRegistrar_##creatorClass s_registrar_##creatorClass; \
    }
```

外部插件推荐通过显式注册函数实现，避免 `DLL_PROCESS_ATTACH` 中持有 loader lock 时调用：

```c
// 插件 DLL 导出注册函数（在 LoadLibrary 后由加载方显式调用）
AICORE_API void AICore_RegisterPlugins(void* builder);
```

```cpp
// 加载方使用方式
HMODULE plugin = LoadLibrary(L"plugins/custom_processor.dll");
auto registerFn = (void(*)(void*))GetProcAddress(plugin, "AICore_RegisterPlugins");
registerFn(&builder);  // 加载方持有 builder 锁的前提下调用，线程安全
```

## Backend 适配层

### BackendConfig

```cpp
struct BackendConfig {
    BackendType type;
    std::string modelPath;
    int inputWidth;
    int inputHeight;
    size_t enginePoolSize;
    bool enableFP16 = true;
    std::string precision;   // "fp32" / "fp16" / "int8"
    Config extraParams;      // 各后端特有参数
};
```

### BackendFactory

```cpp
class BackendFactory {
    static std::unique_ptr<IModelBackend> create(const BackendConfig& cfg);
};
```

### 内存与资源管理

- 统一使用 `std::unique_ptr` 管理对象生命周期
- DLL 层导出 C 接口（`Create` / `Destroy` 配对），避免跨模块 STL/CRT 问题
- `Destroy` 在 DLL 内部调用 `delete`，确保堆一致

```c
// C 接口（跨 DLL 安全）
AICORE_API void* AICore_Create(const char* configPath);
AICORE_API int   AICore_Run(void* handle, const unsigned char* imageData,
                            int width, int height, int channels,
                            char* outJson, int outBufSize);
AICORE_API void  AICore_Destroy(void* handle);
```

### EngineContext

```cpp
struct EngineContext {
    nvinfer1::IExecutionContext* trtContext; // TensorRT 执行上下文
    cudaStream_t stream;                     // 绑定的 CUDA stream
    std::vector<void*> deviceBuffers;        // 预分配的 GPU 显存缓冲区
    size_t bufferBytes;
    bool inUse = false;
};
```

### EnginePool

```cpp
class EnginePool {
public:
    EnginePool(size_t poolSize);
    ~EnginePool();

    // 从池中取一个可用的执行上下文
    EngineContext* acquire(cudaStream_t& stream);

    // 归还到池
    void release(EngineContext* ctx);

    // 扩容
    bool resize(size_t newSize);

private:
    std::mutex mtx_;
    std::queue<EngineContext*> available_;
    std::set<EngineContext*> inUse_;
    std::vector<EngineContext> contexts_; // EngineContext 连续存储
    size_t poolSize_;
};
```

#### IAllocator — 显存/内存统一分配器

```cpp
class IAllocator {
public:
    virtual ~IAllocator() = default;

    // 分配指定类型的内存
    virtual void* allocate(size_t bytes, MemoryType type) = 0;

    // 释放
    virtual void deallocate(void* ptr, MemoryType type) = 0;

    // 当前已分配总量
    virtual size_t allocatedBytes(MemoryType type) const = 0;
};

// 默认实现：CPU 用 malloc/free，GPU 用 cudaMalloc/cudaFree
// Tensor::data 指向分配器管理的内存，Frame 析构时通过 allocId 归还
```

## CUDA Stream 管理

- EnginePool 预创建 `poolSize` 个 CUDA stream，每个 ExecutionContext 绑定一个
- 多线程调用 acquire() 取出空闲上下文及其绑定的 stream
- stream 池大小默认等 CPU 线程数，上限受 GPU 硬件限制（通常 32~128）
- 避免为每个节点创建独立 stream，改为按 ExecutionContext 分配

## 数据结构

```cpp
struct BBox { float x, y, w, h; };

struct NodeResult {
    std::string nodeId;
    std::string label;
    float confidence;
    BBox bbox;
    cv::Mat roi;
    std::map<std::string, double> measurements;
};

struct NodeMetric {
    double latencyMs;
    size_t inputBytes;
    size_t outputBytes;
    StatusCode status;
};

struct Result {
    uint64_t timestamp;
    double totalLatencyMs;
    std::vector<NodeResult> detections;
    std::map<std::string, NodeMetric> nodeMetrics;
    StatusCode status;
    std::string errorMsg;
};
```

## 日志与可观测性

```cpp
// 轻量级日志接口
class ILogger {
public:
    enum Level { kDebug, kInfo, kWarn, kError };
    virtual void log(Level level, const std::string& tag, const std::string& msg) = 0;
};

// Pipeline 内置监控指标收集
struct PipelineMetrics {
    std::map<std::string, NodeMetric> nodeLatencies;
    double gpuUtilPercent;
    size_t gpuMemUsedMB;
    size_t enginePoolQueueDepth;
};
```

## 多线程并发模型

- Pipeline::run() 可多线程并发调用（多相机场景）
- 每个线程从 EnginePool 取出独立 ExecutionContext
- 共享只读资源（模型权重）无锁访问
- Mutex 仅在 resize 引擎池时加锁

### 线程池

```cpp
class ThreadPool {
public:
    explicit ThreadPool(const ThreadPoolConfig& cfg);
    ~ThreadPool();

    // 提交任务，返回 future 获取结果
    template<class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    // 等待所有已提交任务完成
    void waitAll();

    // 停止所有工作线程（拒绝新任务）
    void stop();
};

struct ThreadPoolConfig {
    size_t workerCount = std::thread::hardware_concurrency();  // 默认等于 CPU 核心数
    size_t maxQueueSize = 1024;
    bool enablePriority = false;  // 是否启用优先级队列
};
```

- 线程池用于节点级并行执行（同一拓扑层的节点）
- 推理节点的 GPU 计算在 CUDA stream 上异步执行，不占用 CPU 线程
- 饱和策略：队列满时阻塞调用方

## 超时与 Watchdog

```cpp
struct TimeoutConfig {
    uint32_t nodeTimeoutMs = 30000;      // 单节点超时 (30s)
    uint32_t pipelineTimeoutMs = 60000;  // 整条流水线超时 (60s)
    uint32_t watchdogIntervalMs = 5000;  // 看门狗心跳间隔
};
```

- 节点执行超时 → 返回 ErrorTimeout，释放对应 CUDA context，创建新 context
- GPU hang 检测 → 看门狗线程定时检查 GPU 状态，超时后重置设备
- 超过 3 次连续 GPU 错误 → 上报 ErrorGpuDevice，停止推理服务

## Warm-up 策略

- 引擎初始化完成后，使用随机输入数据执行 3 次推理
- 预热目的：触发 TensorRT kernel 编译、显存分配、cuDNN heuristics 初始化
- 预热完成后才对外提供服务

## 版本兼容策略

### CUDA 11.8 版本配套（初始开发环境）

| 依赖库 | 版本 |
|--------|------|
| CUDA Toolkit | 11.8 |
| TensorRT | 8.5.3 |
| ONNX Runtime | 1.16.3 |
| LibTorch | 2.1.0+cu118 |
| cuDNN | 8.7.0 |

### CUDA 12.x 版本配套（后续迁移）

通过 CMake 变量控制编译时的 CUDA 版本、TensorRT 版本、LibTorch 版本，引擎代码不做硬编码。

### GPU 兼容

- TensorRT 引擎文件与目标 GPU 硬件绑定
- 部署时需在目标工控机上重新构建 TensorRT 引擎
- 引擎构建逻辑封装在独立工具中，不在运行时执行

## 性能预算（参考目标）

| 指标 | 参考值 |
|------|--------|
| 单帧推理总延迟 (1080P) | < 30ms |
| 多流并联吞吐量 (4路) | > 120 FPS |
| GPU 显存占用 | < 4GB (单模型) |
| CPU 内存占用 | < 500MB |
| 引擎池预热时间 | < 5s |

实际指标取决于具体 GPU 型号和模型复杂度，以上为基于 RTX 3060 + YOLOv8s 的参考目标。

## 测试策略

- 单元测试：Google Test，mock GPU 后端返回可控数据
- 集成测试：使用预构建 TensorRT 引擎在 CI 机器上执行
- DLL 接口测试：模拟 Qt 上位机调用 LoadLibrary + Create/Run/Destroy
- 稳定性测试：72 小时持续运行 + 压力测试

## 目录结构

```
aicore/
├── include/
│   ├── core/        # IProcessor, IModelBackend, IPipeline, Frame, Status
│   ├── pipeline/    # Pipeline, ModelNode, CompositeNode, EnginePool
│   ├── backend/     # TensorRTBackend, ONNXRuntimeBackend
│   ├── preprocess/  # ResizeProcessor, NormalizeProcessor 等
│   ├── postprocess/ # NmsProcessor, SoftmaxProcessor 等
│   └── config/      # ConfigParser, PipelineBuilder
├── src/
├── plugins/
├── models/
├── tests/
├── samples/qt_integration/
├── CMakeLists.txt
└── config.json
```
