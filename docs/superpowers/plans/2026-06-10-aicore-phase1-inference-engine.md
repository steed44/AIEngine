# Phase 1: 推理核心 DLL 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可独立工作的推理引擎核心 DLL，支持 JSON 配置驱动的多模型串并联 DAG 编排执行，通过 C API 供 Qt 上位机调用。

**Architecture:** Pipeline + Strategy 设计模式。IProcessor 为统一节点接口，IPipeline 管理 DAG 拓扑执行，IModelBackend 抽象推理后端。ConfigParser 解析 JSON → PipelineConfig，PipelineBuilder 构建 DAG。所有对象通过 unique_ptr 管理，DLL 层导出 C 接口 Create/Destroy 配对。

**Tech Stack:** VS2022, C++17, OpenCV 4.7.0, nlohmann-json (单头文件), Google Test, CUDA 11.8, TensorRT 8.5.3

**Spec:** `docs/superpowers/specs/2026-06-10-aicore-engine-design.md`

---

## 文件结构

```
D:\w\AIEngine\aicore/
├── CMakeLists.txt
├── third_party/
│   └── json.hpp                    # nlohmann-json 单头文件
├── include/
│   ├── core/
│   │   ├── types.h                 # Status, DataType, MemoryType, Tensor, BBox, NodeResult, Result
│   │   ├── frame.h                 # Frame
│   │   ├── processor.h             # IProcessor
│   │   ├── model_backend.h         # IModelBackend, ModelInfo, BackendType
│   │   ├── pipeline.h              # IPipeline
│   │   └── allocator.h             # IAllocator
│   ├── pipeline/
│   │   ├── pipeline_impl.h         # Pipeline 实现
│   │   ├── model_node.h            # ModelNode
│   │   ├── composite_node.h        # CompositeNode
│   │   └── merge_node.h            # MergeNode
│   ├── backend/
│   │   ├── backend_factory.h       # BackendFactory
│   │   ├── tensorrt_backend.h      # TensorRTBackend
│   │   ├── onnxruntime_backend.h   # ONNXRuntimeBackend
│   │   └── libtorch_backend.h      # LibTorchBackend
│   ├── preprocess/
│   │   ├── resize_processor.h
│   │   ├── normalize_processor.h
│   │   └── color_convert_processor.h
│   ├── postprocess/
│   │   ├── nms_processor.h
│   │   ├── softmax_processor.h
│   │   └── label_map_processor.h
│   ├── config/
│   │   ├── config_parser.h         # ConfigParser
│   │   └── pipeline_builder.h      # PipelineBuilder
│   ├── engine/
│   │   ├── engine_pool.h           # EnginePool
│   │   └── thread_pool.h           # ThreadPool
│   └── api/
│       └── aicore_api.h            # C API
├── src/
│   ├── pipeline/
│   │   ├── pipeline_impl.cpp
│   │   ├── model_node.cpp
│   │   ├── composite_node.cpp
│   │   └── merge_node.cpp
│   ├── backend/
│   │   ├── backend_factory.cpp
│   │   ├── tensorrt_backend.cpp
│   │   ├── onnxruntime_backend.cpp
│   │   └── libtorch_backend.cpp
│   ├── preprocess/
│   │   ├── resize_processor.cpp
│   │   ├── normalize_processor.cpp
│   │   └── color_convert_processor.cpp
│   ├── postprocess/
│   │   ├── nms_processor.cpp
│   │   ├── softmax_processor.cpp
│   │   └── label_map_processor.cpp
│   ├── config/
│   │   ├── config_parser.cpp
│   │   └── pipeline_builder.cpp
│   └── api/
│       └── aicore_api.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_types.cpp
│   ├── test_config_parser.cpp
│   ├── test_pipeline_builder.cpp
│   ├── test_pipeline.cpp
│   ├── test_composite_node.cpp
│   ├── test_merge_node.cpp
│   ├── test_thread_pool.cpp
│   └── test_preprocess.cpp
├── config.json                     # 示例流水线配置
└── samples/
    └── qt_integration/
        └── main.cpp
```

---

### Task 1: 项目脚手架 + 基础类型

**Files:**
- Create: `D:\w\AIEngine\aicore\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore\include\core\types.h`
- Create: `D:\w\AIEngine\aicore\tests\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore\tests\test_types.cpp`
- Create: `D:\w\AIEngine\aicore\third_party\json.hpp`（从 https://github.com/nlohmann/json/releases 下载）

- [ ] **Step 1: 下载 nlohmann-json 单头文件**

```bash
# 从 releases 页下载 json.hpp 放入 third_party/
# 或者用 nuget: Install-Package nlohmann.json -Version 3.11.3
```

- [ ] **Step 2: 创建 types.h**

```cpp
#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <opencv2/core.hpp>

namespace aicore {

#ifdef AICORE_EXPORTS
#define AICORE_API __declspec(dllexport)
#else
#define AICORE_API __declspec(dllimport)
#endif

enum class MemoryType { kCPU, kGPU, kPinned };
enum class DataType { kUInt8, kFloat32, kFloat16 };

struct Tensor {
    DataType dtype = DataType::kFloat32;
    std::vector<int64_t> shape;
    MemoryType memory = MemoryType::kCPU;
    void* data = nullptr;
    size_t bytes = 0;
    size_t allocId = 0;
};

enum class StatusCode {
    OK = 0,
    Skip,
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
    StatusCode code = StatusCode::OK;
    std::string message;
    operator bool() const { return code == StatusCode::OK; }
};

struct BBox {
    float x = 0, y = 0, w = 0, h = 0;
};

struct NodeResult {
    std::string nodeId;
    std::string label;
    float confidence = 0;
    BBox bbox;
    cv::Mat roi;
    std::map<std::string, double> measurements;
};

struct NodeMetric {
    double latencyMs = 0;
    size_t inputBytes = 0;
    size_t outputBytes = 0;
    StatusCode status = StatusCode::OK;
};

struct Result {
    uint64_t timestamp = 0;
    double totalLatencyMs = 0;
    std::vector<NodeResult> detections;
    std::map<std::string, NodeMetric> nodeMetrics;
    StatusCode status = StatusCode::OK;
    std::string errorMsg;
};

} // namespace aicore
```

- [ ] **Step 3: 创建测试 test_types.cpp**

```cpp
#include <gtest/gtest.h>
#include "core/types.h"

using namespace aicore;

TEST(TypesTest, StatusOk) {
    Status s;
    EXPECT_TRUE(s);
    EXPECT_EQ(s.code, StatusCode::OK);
}

TEST(TypesTest, StatusError) {
    Status s{StatusCode::ErrorInternal, "test error"};
    EXPECT_FALSE(s);
    EXPECT_EQ(s.message, "test error");
}

TEST(TypesTest, TensorDefaults) {
    Tensor t;
    EXPECT_EQ(t.dtype, DataType::kFloat32);
    EXPECT_EQ(t.memory, MemoryType::kCPU);
    EXPECT_EQ(t.data, nullptr);
}

TEST(TypesTest, ResultDefaults) {
    Result r;
    EXPECT_EQ(r.timestamp, 0);
    EXPECT_TRUE(r.detections.empty());
    EXPECT_EQ(r.status, StatusCode::OK);
}
```

- [ ] **Step 4: 创建 CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(aicore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV 4.7.0 REQUIRED)

# 源文件目录
set(AICORE_SOURCES
    src/api/aicore_api.cpp
    # ... 后续任务逐步添加
)

set(AICORE_HEADERS
    include/core/types.h
    # ... 后续任务逐步添加
)

# 添加第三方库
set(NLOHMANN_JSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# 主库
add_library(aicore SHARED ${AICORE_SOURCES} ${AICORE_HEADERS})
target_include_directories(aicore PUBLIC include)
target_include_directories(aicore PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aicore PRIVATE ${OpenCV_LIBS})
target_compile_definitions(aicore PRIVATE AICORE_EXPORTS)

# 测试
enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 5: 创建 tests/CMakeLists.txt**

```cmake
find_package(GTest REQUIRED)

add_executable(aicore_tests
    test_types.cpp
)

target_include_directories(aicore_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(aicore_tests PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(aicore_tests PRIVATE GTest::GTest GTest::Main ${OpenCV_LIBS})
add_test(NAME aicore_tests COMMAND aicore_tests)
```

- [ ] **Step 6: 构建并运行测试**

Run: `cd D:\w\AIEngine\aicore && mkdir build; cd build; cmake .. -DCMAKE_PREFIX_PATH="path/to/opencv;path/to/gtest"`
Run: `cmake --build . --config Debug`
Run: `ctest --output-on-failure`
Expected: 4 tests pass

---

### Task 2: Frame + IProcessor + IPipeline + IAllocator

**Files:**
- Create: `D:\w\AIEngine\aicore\include\core\frame.h`
- Create: `D:\w\AIEngine\aicore\include\core\processor.h`
- Create: `D:\w\AIEngine\aicore\include\core\pipeline.h`
- Create: `D:\w\AIEngine\aicore\include\core\allocator.h`
- Create: `D:\w\AIEngine\aicore\tests\test_processor.cpp`

- [ ] **Step 1: 创建 frame.h**

```cpp
#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <vector>
#include <map>

namespace aicore {

struct Frame {
    uint64_t id = 0;
    cv::Mat image;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<Tensor> gpuTensors;
    std::map<std::string, NodeResult> nodeResults;
    std::map<std::string, Status> nodeStatuses;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 processor.h**

```cpp
#pragma once
#include "core/frame.h"
#include <nlohmann/json.hpp>
#include <string>

namespace aicore {

using Config = nlohmann::json;

class IProcessor {
public:
    virtual ~IProcessor() = default;
    virtual std::string name() const = 0;
    virtual Status init(const Config& config) = 0;
    virtual Status process(const Frame& input, Frame& output) = 0;
    virtual Status destroy() = 0;
};

} // namespace aicore
```

- [ ] **Step 3: 创建 pipeline.h**

```cpp
#pragma once
#include "core/processor.h"
#include "config/config_parser.h"

namespace aicore {

class IPipeline {
public:
    virtual ~IPipeline() = default;
    virtual Status build(const PipelineConfig& config) = 0;
    virtual Status run(Frame& frame) = 0;
};

} // namespace aicore
```

- [ ] **Step 4: 创建 allocator.h**

```cpp
#pragma once
#include "core/types.h"

namespace aicore {

class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* allocate(size_t bytes, MemoryType type) = 0;
    virtual void deallocate(void* ptr, MemoryType type) = 0;
    virtual size_t allocatedBytes(MemoryType type) const = 0;
};

// 注：IAllocator 在 Phase 1 仅定义接口，不接入使用
// Phase 2 实现 TensorRT 后端时再通过 EnginePool 接入

} // namespace aicore
```

- [ ] **Step 5: 写入测试并运行**

```cpp
#include <gtest/gtest.h>
#include "core/processor.h"
#include "core/frame.h"

using namespace aicore;

class TestProcessor : public IProcessor {
    std::string name_;
public:
    TestProcessor(const std::string& name) : name_(name) {}
    std::string name() const override { return name_; }
    Status init(const Config&) override { return Status{}; }
    Status process(const Frame& input, Frame& output) override {
        output.id = input.id;
        output.height = input.height;
        return Status{};
    }
    Status destroy() override { return Status{}; }
};

TEST(ProcessorTest, BasicLifecycle) {
    TestProcessor proc("test");
    EXPECT_EQ(proc.name(), "test");
    EXPECT_TRUE(proc.init(Config::object()));
    
    Frame in, out;
    in.id = 42;
    in.height = 480;
    EXPECT_TRUE(proc.process(in, out));
    EXPECT_EQ(out.id, 42);
    EXPECT_TRUE(proc.destroy());
}
```

Update tests/CMakeLists.txt to add test_processor.cpp.

---

### Task 3: IModelBackend + BackendFactory

**Files:**
- Create: `D:\w\AIEngine\aicore\include\core\model_backend.h`
- Create: `D:\w\AIEngine\aicore\include\backend\backend_factory.h`
- Create: `D:\w\AIEngine\aicore\src\backend\backend_factory.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_backend_factory.cpp`

- [ ] **Step 1: 创建 model_backend.h**

```cpp
#pragma once
#include "core/processor.h"
#include <vector>

namespace aicore {

enum class BackendType { kTensorRT, kONNXRuntime, kLibTorch };

struct IOInfo {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype;
};

struct ModelInfo {
    std::string name;
    std::vector<IOInfo> inputs;
    std::vector<IOInfo> outputs;
    BackendType backend;
    std::string modelPath;
};

// 约定: IModelBackend 的 process() 应返回 ErrorInternal
// 外部通过 ModelNode::process() 调用，后者编排预处理→infer→后处理
class IModelBackend : public IProcessor {
public:
    virtual ModelInfo modelInfo() const = 0;
    virtual Status infer(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) = 0;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 backend_factory.h**

```cpp
#pragma once
#include "core/model_backend.h"
#include <memory>

namespace aicore {

struct BackendConfig {
    BackendType type;
    std::string modelPath;
    int inputWidth = 640;
    int inputHeight = 640;
    size_t enginePoolSize = 3;
    bool enableFP16 = true;
    std::string precision = "fp16";
    Config extraParams;
};

class BackendFactory {
public:
    static std::unique_ptr<IModelBackend> create(const BackendConfig& cfg);
};

} // namespace aicore
```

- [ ] **Step 3: 创建后端头文件占位**

```cpp
// include/backend/tensorrt_backend.h
#pragma once
#include "core/model_backend.h"
namespace aicore {
class TensorRTBackend : public IModelBackend {
public:
    TensorRTBackend() = default;
    ~TensorRTBackend() override;
    std::string name() const override;
    Status init(const Config& cfg) override;
    Status process(const Frame& input, Frame& output) override;
    Status infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    ModelInfo modelInfo() const override;
    Status destroy() override;
};
} // namespace aicore
```

```cpp
// include/backend/onnxruntime_backend.h (同 tensorrt_backend.h 结构，类名 ONNXRuntimeBackend)
```

```cpp
// include/backend/libtorch_backend.h (同 tensorrt_backend.h 结构，类名 LibTorchBackend)
```

- [ ] **Step 4: 创建 backend_factory.cpp**

```cpp
#include "backend/backend_factory.h"
#include "backend/tensorrt_backend.h"
#include "backend/onnxruntime_backend.h"
#include "backend/libtorch_backend.h"

namespace aicore {

std::unique_ptr<IModelBackend> BackendFactory::create(const BackendConfig& cfg) {
    switch (cfg.type) {
        case BackendType::kTensorRT:
            return std::make_unique<TensorRTBackend>();
        case BackendType::kONNXRuntime:
            return std::make_unique<ONNXRuntimeBackend>();
        case BackendType::kLibTorch:
            return std::make_unique<LibTorchBackend>();
    }
    return nullptr;
}

} // namespace aicore
```

- [ ] **Step 5: 测试并运行**

```cpp
#include <gtest/gtest.h>
#include "backend/backend_factory.h"

using namespace aicore;

TEST(BackendFactoryTest, CreateTensorRT) {
    BackendConfig cfg;
    cfg.type = BackendType::kTensorRT;
    auto backend = BackendFactory::create(cfg);
    // TensorRTBackend 尚未实现，目前是空指针
    // 当后端实现后，此处应 EXPECT_NE(backend, nullptr);
}
```

---

### Task 4: ConfigParser — JSON 配置解析

**Files:**
- Create: `D:\w\AIEngine\aicore\include\config\config_parser.h`
- Create: `D:\w\AIEngine\aicore\src\config\config_parser.cpp`
- Create: `D:\w\AIEngine\aicore\config.json`
- Create: `D:\w\AIEngine\aicore\tests\test_config_parser.cpp`

- [ ] **Step 1: 创建 config_parser.h**

```cpp
#pragma once
#include "core/processor.h"
#include <string>
#include <vector>

namespace aicore {

struct NodeConfig {
    std::string id;
    std::string type;       // model / preprocess / postprocess / merge / composite
    std::string name;
    std::string backend;    // 仅 type=model
    std::string modelPath;
    int inputWidth = 0;
    int inputHeight = 0;
    std::vector<Config> preprocessSteps;
    std::vector<Config> postprocessSteps;
    Config params;
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

class ConfigParser {
public:
    static PipelineConfig parseFile(const std::string& path);
    static PipelineConfig parseString(const std::string& jsonStr);
    static bool validate(const PipelineConfig& config, std::string* errorOut = nullptr);
};

} // namespace aicore
```

- [ ] **Step 2: 创建 config_parser.cpp**

```cpp
#include "config/config_parser.h"
#include <fstream>
#include <set>

namespace aicore {

PipelineConfig ConfigParser::parseString(const std::string& jsonStr) {
    auto j = Config::parse(jsonStr);
    PipelineConfig config;
    config.name = j["pipeline"]["name"];

    for (auto& n : j["pipeline"]["nodes"]) {
        NodeConfig nc;
        nc.id = n["id"];
        nc.type = n.value("type", "model");
        nc.name = n.value("name", nc.id);
        nc.backend = n.value("backend", "tensorrt");
        nc.modelPath = n.value("model", "");
        nc.inputWidth = n["input"].value("width", 640);
        nc.inputHeight = n["input"].value("height", 640);
        nc.params = n.value("params", Config::object());
        nc.required = n.value("required", false);
        nc.enginePoolSize = n.value("engine_pool_size", 3);

        if (n.contains("preprocess")) {
            for (auto& p : n["preprocess"]) {
                nc.preprocessSteps.push_back(p);
            }
        }
        if (n.contains("postprocess")) {
            for (auto& p : n["postprocess"]) {
                nc.postprocessSteps.push_back(p);
            }
        }
        config.nodes.push_back(nc);
    }

    for (auto& e : j["pipeline"]["edges"]) {
        EdgeConfig ec;
        ec.from = e["from"];
        ec.to = e["to"];
        config.edges.push_back(ec);
    }

    return config;
}

PipelineConfig ConfigParser::parseFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    return parseString(content);
}

bool ConfigParser::validate(const PipelineConfig& config, std::string* errorOut) {
    // 1. 检查所有节点 id 唯一
    std::set<std::string> ids;
    for (auto& n : config.nodes) {
        if (ids.count(n.id)) {
            if (errorOut) *errorOut = "Duplicate node id: " + n.id;
            return false;
        }
        ids.insert(n.id);
    }

    // 2. 检查边引用的节点存在
    for (auto& e : config.edges) {
        if (!ids.count(e.from) && e.from != "input") {
            if (errorOut) *errorOut = "Edge refers to unknown node: " + e.from;
            return false;
        }
        if (!ids.count(e.to) && e.to != "output") {
            if (errorOut) *errorOut = "Edge refers to unknown node: " + e.to;
            return false;
        }
    }

    // 3. 检查 model 类型节点有后端
    for (auto& n : config.nodes) {
        if (n.type == "model" && n.modelPath.empty()) {
            if (errorOut) *errorOut = "Model node " + n.id + " has no model path";
            return false;
        }
    }

    return true;
}

} // namespace aicore
```

- [ ] **Step 3: 创建 config.json**

```json
{
  "pipeline": {
    "name": "示例缺陷检测流水线",
    "nodes": [
      {
        "id": "detector",
        "type": "model",
        "name": "缺陷检测",
        "backend": "tensorrt",
        "model": "models/defect_detector.engine",
        "input": { "width": 640, "height": 640 },
        "preprocess": [
          {"type": "resize", "width": 640, "height": 640},
          {"type": "normalize", "mean": [0,0,0], "std": [1,1,1]}
        ],
        "postprocess": [
          {"type": "nms", "iou_threshold": 0.5, "conf_threshold": 0.3}
        ],
        "required": true,
        "engine_pool_size": 3
      }
    ],
    "edges": [
      {"from": "input", "to": "detector"},
      {"from": "detector", "to": "output"}
    ]
  }
}
```

- [ ] **Step 4: 测试并运行**

```cpp
#include <gtest/gtest.h>
#include "config/config_parser.h"

using namespace aicore;

TEST(ConfigParserTest, ParseSimplePipeline) {
    std::string json = R"({
        "pipeline": {
            "name": "test",
            "nodes": [{
                "id": "n1", "type": "model", "name": "detector",
                "backend": "tensorrt", "model": "test.engine",
                "input": {"width": 640, "height": 640}
            }],
            "edges": [
                {"from": "input", "to": "n1"},
                {"from": "n1", "to": "output"}
            ]
        }
    })";

    auto cfg = ConfigParser::parseString(json);
    EXPECT_EQ(cfg.name, "test");
    EXPECT_EQ(cfg.nodes.size(), 1);
    EXPECT_EQ(cfg.nodes[0].id, "n1");
    EXPECT_EQ(cfg.edges.size(), 2);
}

TEST(ConfigParserTest, ValidateDuplicateId) {
    std::string json = R"({
        "pipeline": {
            "name": "test",
            "nodes": [
                {"id": "n1", "type": "model", "input": {"width": 640, "height": 640}},
                {"id": "n1", "type": "model", "input": {"width": 640, "height": 640}}
            ],
            "edges": []
        }
    })";

    auto cfg = ConfigParser::parseString(json);
    std::string err;
    EXPECT_FALSE(ConfigParser::validate(cfg, &err));
    EXPECT_TRUE(err.find("Duplicate") != std::string::npos);
}
```

---

### Task 5: ThreadPool + EnginePool

**Files:**
- Create: `D:\w\AIEngine\aicore\include\engine\thread_pool.h`
- Create: `D:\w\AIEngine\aicore\src\engine\thread_pool.cpp`
- Create: `D:\w\AIEngine\aicore\include\engine\engine_pool.h`
- Create: `D:\w\AIEngine\aicore\src\engine\engine_pool.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_thread_pool.cpp`

- [ ] **Step 1: 创建 thread_pool.h**

```cpp
#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

namespace aicore {

struct ThreadPoolConfig {
    size_t workerCount = std::thread::hardware_concurrency();
    size_t maxQueueSize = 1024;
    bool enablePriority = false;
};

class ThreadPool {
public:
    explicit ThreadPool(const ThreadPoolConfig& cfg = ThreadPoolConfig{});
    ~ThreadPool();

    template<class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.size() >= maxQueueSize_) {
                throw std::runtime_error("ThreadPool queue full");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cond_.notify_one();
        return result;
    }

    void waitAll();
    void stop();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<bool> stopping_{false};
    size_t maxQueueSize_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 thread_pool.cpp**

```cpp
#include "engine/thread_pool.h"

namespace aicore {

ThreadPool::ThreadPool(const ThreadPoolConfig& cfg)
    : maxQueueSize_(cfg.maxQueueSize) {
    for (size_t i = 0; i < cfg.workerCount; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cond_.wait(lock, [this]() {
                        return stopping_ || !tasks_.empty();
                    });
                    if (stopping_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::stop() {
    stopping_ = true;
    cond_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::waitAll() {
    // 简单版：提交一个空任务并等待其完成，实际需要更完善的机制
    auto f = submit([]() {});
    f.get();
}

} // namespace aicore
```

- [ ] **Step 3: 创建 engine_pool.h（先做接口，具体 CUDA 实现在后端实现后补充）**

```cpp
#pragma once
#include "core/types.h"

namespace aicore {

struct EngineContext {
    void* trtContext = nullptr;    // nvinfer1::IExecutionContext*
    void* stream = nullptr;        // cudaStream_t
    std::vector<void*> deviceBuffers;
    size_t bufferBytes = 0;
    bool inUse = false;
};

class EnginePool {
public:
    EnginePool(size_t poolSize);
    ~EnginePool();
    EngineContext* acquire(void*& stream);
    void release(EngineContext* ctx);
    bool resize(size_t newSize);
    size_t size() const { return poolSize_; }

private:
    std::mutex mtx_;
    std::vector<EngineContext> contexts_;
    std::queue<EngineContext*> available_;
    size_t poolSize_;
};

} // namespace aicore
```

- [ ] **Step 4: 创建 engine_pool.cpp**

```cpp
#include "engine/engine_pool.h"

namespace aicore {

EnginePool::EnginePool(size_t poolSize) : poolSize_(poolSize) {
    contexts_.resize(poolSize);
    for (auto& ctx : contexts_) {
        available_.push(&ctx);
    }
}

EnginePool::~EnginePool() {
    // CUDA 资源释放留到后端实现时补充
}

EngineContext* EnginePool::acquire(void*& stream) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (available_.empty()) return nullptr;
    auto ctx = available_.front();
    available_.pop();
    ctx->inUse = true;
    stream = ctx->stream;
    return ctx;
}

void EnginePool::release(EngineContext* ctx) {
    std::lock_guard<std::mutex> lock(mtx_);
    ctx->inUse = false;
    available_.push(ctx);
}

bool EnginePool::resize(size_t newSize) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (newSize <= poolSize_) return true;
    size_t oldSize = contexts_.size();
    contexts_.resize(newSize);
    for (size_t i = oldSize; i < newSize; ++i) {
        available_.push(&contexts_[i]);
    }
    poolSize_ = newSize;
    return true;
}

} // namespace aicore
```

- [ ] **Step 5: 测试 ThreadPool**

```cpp
#include <gtest/gtest.h>
#include "engine/thread_pool.h"
#include <atomic>

using namespace aicore;

TEST(ThreadPoolTest, SubmitAndExecute) {
    ThreadPool pool({2});
    std::atomic<int> counter{0};
    
    auto f1 = pool.submit([&]() { counter++; return 1; });
    auto f2 = pool.submit([&]() { counter++; return 2; });
    
    EXPECT_EQ(f1.get(), 1);
    EXPECT_EQ(f2.get(), 2);
    EXPECT_EQ(counter.load(), 2);
}

TEST(ThreadPoolTest, ParallelExecution) {
    ThreadPool pool({4});
    std::atomic<int> sum{0};
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([&, i]() {
            sum.fetch_add(i);
        }));
    }
    
    for (auto& f : futures) f.get();
    EXPECT_EQ(sum.load(), 4950);  // 0+1+...+99
}
```

---

### Task 6: PipelineBuilder

**Files:**
- Create: `D:\w\AIEngine\aicore\include\config\pipeline_builder.h`
- Create: `D:\w\AIEngine\aicore\src\config\pipeline_builder.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_pipeline_builder.cpp`

- [ ] **Step 1: 创建 pipeline_builder.h**

```cpp
#pragma once
#include "core/pipeline.h"
#include "config/config_parser.h"
#include <memory>
#include <unordered_map>
#include <functional>

namespace aicore {

class PipelineBuilder {
public:
    using CreatorFunc = std::function<std::unique_ptr<IProcessor>(const Config&)>;

    void registerNodeType(const std::string& type, CreatorFunc creator);

    std::unique_ptr<IPipeline> build(const PipelineConfig& config);

    static PipelineBuilder& globalInstance();

private:
    std::unordered_map<std::string, CreatorFunc> registry_;
    std::mutex registryMutex_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 pipeline_builder.cpp（先做框架，注册基本节点类型）**

```cpp
#include "config/pipeline_builder.h"

namespace aicore {

void PipelineBuilder::registerNodeType(const std::string& type, CreatorFunc creator) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    registry_[type] = std::move(creator);
}

PipelineBuilder& PipelineBuilder::globalInstance() {
    static PipelineBuilder instance;
    return instance;
}

std::unique_ptr<IPipeline> PipelineBuilder::build(const PipelineConfig& config) {
    // 校验配置
    std::string error;
    if (!ConfigParser::validate(config, &error)) {
        throw std::runtime_error("Invalid pipeline config: " + error);
    }

    auto pipeline = std::make_unique<Pipeline>();

    // 遍历配置创建节点（前提：节点类型已注册）
    for (auto& nodeCfg : config.nodes) {
        auto it = registry_.find(nodeCfg.type);
        if (it == registry_.end()) {
            throw std::runtime_error("Unknown node type: " + nodeCfg.type);
        }
        auto processor = it->second(nodeCfg.params);
        pipeline->addNode(nodeCfg.id, std::move(processor), nodeCfg.required);
    }

    // 建立连接
    for (auto& edge : config.edges) {
        // 查找节点索引（简化版：通过节点 ID 查找索引）
        auto findIdx = [&](const std::string& id) -> size_t {
            for (size_t i = 0; i < config.nodes.size(); ++i) {
                if (config.nodes[i].id == id) return i;
            }
            return static_cast<size_t>(-1);
        };
        size_t fromIdx = findIdx(edge.from);
        size_t toIdx = findIdx(edge.to);
        if (fromIdx != static_cast<size_t>(-1) && toIdx != static_cast<size_t>(-1)) {
            pipeline->addEdge(fromIdx, toIdx);
        }
    }

    auto status = pipeline->build(config);
    if (!status) {
        throw std::runtime_error("Pipeline build failed: " + status.message);
    }

    return pipeline;
}

} // namespace aicore
```

- [ ] **Step 3: 测试**

```cpp
#include <gtest/gtest.h>
#include "config/pipeline_builder.h"
#include "config/config_parser.h"

using namespace aicore;

TEST(PipelineBuilderTest, RegisterAndBuild) {
    auto& builder = PipelineBuilder::globalInstance();
    
    builder.registerNodeType("model", [](const Config&) -> std::unique_ptr<IProcessor> {
        return nullptr;  // 暂不实现
    });

    std::string json = R"({
        "pipeline": {
            "name": "test",
            "nodes": [{
                "id": "n1", "type": "model",
                "input": {"width": 640, "height": 640}
            }],
            "edges": [
                {"from": "input", "to": "n1"},
                {"from": "n1", "to": "output"}
            ]
        }
    })";

    auto cfg = ConfigParser::parseString(json);
    EXPECT_THROW(builder.build(cfg), std::runtime_error); // 暂未注册 model 节点
}
```

---

### Task 7: Pipeline DAG 执行引擎

**Files:**
- Create: `D:\w\AIEngine\aicore\include\pipeline\pipeline_impl.h`
- Create: `D:\w\AIEngine\aicore\src\pipeline\pipeline_impl.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_pipeline.cpp`

- [ ] **Step 1: 创建 pipeline_impl.h**

```cpp
#pragma once
#include "core/pipeline.h"
#include "config/config_parser.h"
#include "engine/thread_pool.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>

namespace aicore {

struct NodeDef {
    std::string id;
    std::unique_ptr<IProcessor> processor;
    std::vector<size_t> inputIndices;   // 上游节点索引
    std::vector<size_t> outputIndices;  // 下游节点索引
    bool required = false;
};

class Pipeline : public IPipeline {
public:
    Pipeline();
    ~Pipeline() override;

    Status build(const PipelineConfig& config) override;
    Status run(Frame& frame) override;

    // 添加节点和边（供 PipelineBuilder 调用）
    size_t addNode(const std::string& id, std::unique_ptr<IProcessor> processor, bool required = false);
    void addEdge(size_t fromIdx, size_t toIdx);

private:
    std::vector<NodeDef> nodes_;
    std::vector<size_t> execOrder_;       // 拓扑排序缓存
    std::unique_ptr<ThreadPool> pool_;
    bool built_ = false;

    // 拓扑排序（Kahn 算法）
    bool topologicalSort(std::vector<size_t>& order);
};

} // namespace aicore
```

- [ ] **Step 2: 创建 pipeline_impl.cpp**

```cpp
#include "pipeline/pipeline_impl.h"
#include <queue>
#include <algorithm>

namespace aicore {

Pipeline::Pipeline()
    : pool_(std::make_unique<ThreadPool>(ThreadPoolConfig{})) {}

Pipeline::~Pipeline() = default;

size_t Pipeline::addNode(const std::string& id, std::unique_ptr<IProcessor> processor, bool required) {
    size_t idx = nodes_.size();
    nodes_.push_back({id, std::move(processor), {}, {}, required});
    return idx;
}

void Pipeline::addEdge(size_t fromIdx, size_t toIdx) {
    nodes_[fromIdx].outputIndices.push_back(toIdx);
    nodes_[toIdx].inputIndices.push_back(fromIdx);
}

bool Pipeline::topologicalSort(std::vector<size_t>& order) {
    std::vector<size_t> inDegree(nodes_.size(), 0);
    for (auto& node : nodes_) {
        for (auto out : node.outputIndices) {
            inDegree[out]++;
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (inDegree[i] == 0) q.push(i);
    }

    order.clear();
    while (!q.empty()) {
        size_t idx = q.front(); q.pop();
        order.push_back(idx);
        for (auto out : nodes_[idx].outputIndices) {
            if (--inDegree[out] == 0) q.push(out);
        }
    }

    return order.size() == nodes_.size();
}

Status Pipeline::build(const PipelineConfig& config) {
    // 拓扑排序
    if (!topologicalSort(execOrder_)) {
        return Status{StatusCode::ErrorConfigParse, "Pipeline contains cycle"};
    }
    built_ = true;
    return Status{};
}

Status Pipeline::run(Frame& frame) {
    if (!built_) {
        return Status{StatusCode::ErrorInternal, "Pipeline not built"};
    }

    // 按拓扑排序逐层执行
    // 简单版：串行遍历执行（后续优化为分层并行）
    for (size_t idx : execOrder_) {
        auto& node = nodes_[idx];

        // 检查上游是否全部失败
        bool upstreamFailed = false;
        for (size_t inIdx : node.inputIndices) {
            auto it = frame.nodeStatuses.find(nodes_[inIdx].id);
            if (it != frame.nodeStatuses.end() && it->second.code != StatusCode::OK) {
                upstreamFailed = true;
                break;
            }
        }

        if (upstreamFailed && !node.required) {
            frame.nodeStatuses[node.id] = Status{StatusCode::Skip, "Upstream failed"};
            continue;
        }

        // 执行节点
        Frame nodeOutput;
        auto status = node.processor->process(frame, nodeOutput);

        // 合并输出到主 Frame
        for (auto& [k, v] : nodeOutput.nodeResults) {
            frame.nodeResults[k] = std::move(v);
        }
        frame.nodeStatuses[node.id] = status;

        if (status.code != StatusCode::OK && node.required) {
            return status;
        }
    }

    return Status{};
}

} // namespace aicore
```

- [ ] **Step 3: 测试（使用模拟节点）**

```cpp
#include <gtest/gtest.h>
#include "pipeline/pipeline_impl.h"
#include "core/processor.h"

using namespace aicore;

class MockProcessor : public IProcessor {
    std::string name_;
    StatusCode returnCode_;
public:
    MockProcessor(const std::string& name, StatusCode code = StatusCode::OK)
        : name_(name), returnCode_(code) {}
    std::string name() const override { return name_; }
    Status init(const Config&) override { return Status{}; }
    Status process(const Frame& input, Frame& output) override {
        output.id = input.id;
        NodeResult nr;
        nr.nodeId = name_;
        nr.confidence = 0.95f;
        output.nodeResults[name_] = nr;
        return Status{returnCode_, ""};
    }
    Status destroy() override { return Status{}; }
};

TEST(PipelineTest, SimpleSerial) {
    Pipeline pipeline;
    auto n1 = pipeline.addNode("n1", std::make_unique<MockProcessor>("n1"));
    auto n2 = pipeline.addNode("n2", std::make_unique<MockProcessor>("n2"));
    pipeline.addEdge(n1, n2);

    PipelineConfig cfg;
    cfg.name = "test";
    ASSERT_TRUE(pipeline.build(cfg));

    Frame input;
    input.id = 1;
    Frame output;
    ASSERT_TRUE(pipeline.run(input));
    ASSERT_EQ(input.nodeResults.count("n2"), 1);
}

TEST(PipelineTest, UpstreamFailureSkip) {
    Pipeline pipeline;
    auto n1 = pipeline.addNode("n1", std::make_unique<MockProcessor>("n1", StatusCode::ErrorInternal));
    auto n2 = pipeline.addNode("n2", std::make_unique<MockProcessor>("n2"));
    pipeline.addEdge(n1, n2);

    PipelineConfig cfg;
    ASSERT_TRUE(pipeline.build(cfg));

    Frame input;
    input.id = 1;
    ASSERT_TRUE(pipeline.run(input));
    EXPECT_EQ(input.nodeStatuses["n1"].code, StatusCode::ErrorInternal);
    EXPECT_EQ(input.nodeStatuses["n2"].code, StatusCode::Skip);
}
```

---

### Task 8: ModelNode + CompositeNode + MergeNode

**Files:**
- Create: `D:\w\AIEngine\aicore\include\pipeline\model_node.h`
- Create: `D:\w\AIEngine\aicore\src\pipeline\model_node.cpp`
- Create: `D:\w\AIEngine\aicore\include\pipeline\composite_node.h`
- Create: `D:\w\AIEngine\aicore\src\pipeline\composite_node.cpp`
- Create: `D:\w\AIEngine\aicore\include\pipeline\merge_node.h`
- Create: `D:\w\AIEngine\aicore\src\pipeline\merge_node.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_composite_node.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_merge_node.cpp`

- [ ] **Step 1: 创建 model_node.h**

```cpp
#pragma once
#include "core/processor.h"
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <memory>

namespace aicore {

// ModelNode 编排：预处理 → IModelBackend::infer → 后处理
class ModelNode : public IProcessor {
public:
    explicit ModelNode(const Config& config);
    std::string name() const override;
    Status init(const Config& config) override;
    Status process(const Frame& input, Frame& output) override;
    Status destroy() override;

private:
    std::string name_;
    std::unique_ptr<IModelBackend> backend_;
    std::vector<std::unique_ptr<IProcessor>> preprocessSteps_;
    std::vector<std::unique_ptr<IProcessor>> postprocessSteps_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 model_node.cpp（骨架，preprocess/postprocess 节点后续实作）**

```cpp
#include "pipeline/model_node.h"

namespace aicore {

ModelNode::ModelNode(const Config& config) {
    name_ = config.value("name", config.value("id", "model"));
}

std::string ModelNode::name() const { return name_; }

Status ModelNode::init(const Config& config) {
    // 创建后端
    BackendConfig backendCfg;
    backendCfg.type = BackendType::kTensorRT;
    backendCfg.modelPath = config.value("model", "");
    backendCfg.inputWidth = config["input"].value("width", 640);
    backendCfg.inputHeight = config["input"].value("height", 640);
    backendCfg.enginePoolSize = config.value("engine_pool_size", 3);

    auto backendType = config.value("backend", "tensorrt");
    if (backendType == "onnxruntime") backendCfg.type = BackendType::kONNXRuntime;
    else if (backendType == "libtorch") backendCfg.type = BackendType::kLibTorch;

    backend_ = BackendFactory::create(backendCfg);
    if (!backend_) {
        return Status{StatusCode::ErrorModelLoad, "Failed to create backend: " + backendType};
    }

    // 初始化后端
    auto status = backend_->init(backendCfg);
    if (!status) return status;

    return Status{};
}

Status ModelNode::process(const Frame& input, Frame& output) {
    Frame current = input;

    // 执行预处理步骤
    for (auto& step : preprocessSteps_) {
        Frame next;
        auto status = step->process(current, next);
        if (!status) return status;
        current = std::move(next);
    }

    // 执行推理
    std::vector<Tensor> inputs = current.gpuTensors;
    std::vector<Tensor> outputs;
    auto status = backend_->infer(inputs, outputs);
    if (!status) return status;

    // 将输出结果写入 Frame
    for (size_t i = 0; i < outputs.size(); ++i) {
        current.gpuTensors.push_back(std::move(outputs[i]));
    }

    // 执行后处理步骤
    for (auto& step : postprocessSteps_) {
        Frame next;
        auto s = step->process(current, next);
        if (!s) return s;
        current = std::move(next);
    }

    output = std::move(current);
    return Status{};
}

Status ModelNode::destroy() {
    if (backend_) return backend_->destroy();
    return Status{};
}

} // namespace aicore
```

- [ ] **Step 3: 创建 composite_node.h**

```cpp
#pragma once
#include "core/processor.h"
#include <vector>
#include <memory>

namespace aicore {

enum class CompositeType { kSerial, kParallel };

class CompositeNode : public IProcessor {
public:
    CompositeNode(const std::string& name, CompositeType type);
    
    void addChild(std::unique_ptr<IProcessor> child);

    std::string name() const override;
    Status init(const Config& config) override;
    Status process(const Frame& input, Frame& output) override;
    Status destroy() override;

private:
    std::string name_;
    CompositeType type_;
    std::vector<std::unique_ptr<IProcessor>> children_;
};

} // namespace aicore
```

- [ ] **Step 4: 创建 composite_node.cpp**

```cpp
#include "pipeline/composite_node.h"

namespace aicore {

CompositeNode::CompositeNode(const std::string& name, CompositeType type)
    : name_(name), type_(type) {}

void CompositeNode::addChild(std::unique_ptr<IProcessor> child) {
    children_.push_back(std::move(child));
}

std::string CompositeNode::name() const { return name_; }

Status CompositeNode::init(const Config& config) {
    for (auto& child : children_) {
        auto status = child->init(config);
        if (!status) return status;
    }
    return Status{};
}

Status CompositeNode::process(const Frame& input, Frame& output) {
    if (type_ == CompositeType::kSerial) {
        Frame current = input;
        for (auto& child : children_) {
            Frame next;
            auto status = child->process(current, next);
            if (!status) return status;
            current = std::move(next);
        }
        output = std::move(current);
    } else {
        // Parallel: 简化为串行，后续用线程池并行
        // Phase 2 中优化为真正并行
        output = input;
        for (auto& child : children_) {
            Frame childOut;
            auto status = child->process(input, childOut);
            if (!status) return status;
            // 合并结果
            for (auto& [k, v] : childOut.nodeResults) {
                output.nodeResults[k] = std::move(v);
            }
        }
    }
    return Status{};
}

Status CompositeNode::destroy() {
    for (auto& child : children_) {
        auto status = child->destroy();
        if (!status) return status;
    }
    return Status{};
}

} // namespace aicore
```

- [ ] **Step 5: 创建 merge_node.h**

```cpp
#pragma once
#include "core/processor.h"
#include <string>

namespace aicore {

enum class MergeStrategy { kUnion, kMaxScore, kAverage };

class MergeNode : public IProcessor {
public:
    explicit MergeNode(const std::string& strategy = "union");
    std::string name() const override;
    Status init(const Config& config) override;
    Status process(const Frame& input, Frame& output) override;
    Status destroy() override;

private:
    std::string name_;
    MergeStrategy strategy_;
    static MergeStrategy parseStrategy(const std::string& s);
};

} // namespace aicore
```

- [ ] **Step 6: 创建 merge_node.cpp**

```cpp
#include "pipeline/merge_node.h"

namespace aicore {

MergeNode::MergeNode(const std::string& strategy)
    : name_("merge"), strategy_(parseStrategy(strategy)) {}

std::string MergeNode::name() const { return name_; }

Status MergeNode::init(const Config& config) {
    name_ = config.value("name", "merge");
    strategy_ = parseStrategy(config.value("strategy", "union"));
    return Status{};
}

MergeStrategy MergeNode::parseStrategy(const std::string& s) {
    if (s == "max_score") return MergeStrategy::kMaxScore;
    if (s == "average") return MergeStrategy::kAverage;
    return MergeStrategy::kUnion;
}

Status MergeNode::process(const Frame& input, Frame& output) {
    output = input;

    // 收集所有上游节点的检测结果
    std::vector<NodeResult> allResults;
    for (auto& [nodeId, result] : input.nodeResults) {
        allResults.push_back(result);
    }

    // 根据策略合并
    switch (strategy_) {
        case MergeStrategy::kUnion:
            // 直接合并所有结果
            for (auto& r : allResults) {
                output.nodeResults["merged_" + r.nodeId] = r;
            }
            break;

        case MergeStrategy::kMaxScore: {
            // 取最高置信度
            NodeResult best;
            for (auto& r : allResults) {
                if (r.confidence > best.confidence) best = r;
            }
            if (!allResults.empty()) {
                output.nodeResults["merged_best"] = best;
            }
            break;
        }

        case MergeStrategy::kAverage: {
            // 平均置信度（简化为合并）
            double avgConf = 0;
            for (auto& r : allResults) avgConf += r.confidence;
            if (!allResults.empty()) avgConf /= allResults.size();

            NodeResult merged;
            merged.nodeId = "merged_avg";
            merged.confidence = static_cast<float>(avgConf);
            if (!allResults.empty()) merged.label = allResults[0].label;
            output.nodeResults["merged_avg"] = merged;
            break;
        }
    }

    return Status{};
}

Status MergeNode::destroy() { return Status{}; }

} // namespace aicore
```

- [ ] **Step 7: 测试 CompositeNode**

```cpp
#include <gtest/gtest.h>
#include "pipeline/composite_node.h"
#include "core/processor.h"

using namespace aicore;

class IncrementProcessor : public IProcessor {
    int* counter_;
public:
    explicit IncrementProcessor(int* c) : counter_(c) {}
    std::string name() const override { return "inc"; }
    Status init(const Config&) override { return Status{}; }
    Status process(const Frame& input, Frame& output) override {
        (*counter_)++;
        output = input;
        return Status{};
    }
    Status destroy() override { return Status{}; }
};

TEST(CompositeNodeTest, SerialExecution) {
    int count = 0;
    auto composite = std::make_unique<CompositeNode>("serial", CompositeType::kSerial);
    composite->addChild(std::make_unique<IncrementProcessor>(&count));
    composite->addChild(std::make_unique<IncrementProcessor>(&count));
    composite->addChild(std::make_unique<IncrementProcessor>(&count));

    ASSERT_TRUE(composite->init(Config::object()));
    Frame in, out;
    ASSERT_TRUE(composite->process(in, out));
    EXPECT_EQ(count, 3);
}

TEST(MergeNodeTest, UnionStrategy) {
    MergeNode merge("union");
    ASSERT_TRUE(merge.init(Config::object()));

    Frame in;
    NodeResult r1; r1.nodeId = "a"; r1.confidence = 0.9f;
    NodeResult r2; r2.nodeId = "b"; r2.confidence = 0.8f;
    in.nodeResults["a"] = r1;
    in.nodeResults["b"] = r2;

    Frame out;
    ASSERT_TRUE(merge.process(in, out));
    EXPECT_EQ(out.nodeResults.size(), 2);
}
```

---

### Task 9: 预处理/后处理节点

**Files:**
- Create: `D:\w\AIEngine\aicore\include\preprocess\resize_processor.h`
- Create: `D:\w\AIEngine\aicore\src\preprocess\resize_processor.cpp`
- Create: `D:\w\AIEngine\aicore\include\preprocess\normalize_processor.h`
- Create: `D:\w\AIEngine\aicore\src\preprocess\normalize_processor.cpp`
- Create: `D:\w\AIEngine\aicore\include\preprocess\color_convert_processor.h`
- Create: `D:\w\AIEngine\aicore\src\preprocess\color_convert_processor.cpp`
- Create: `D:\w\AIEngine\aicore\include\postprocess\nms_processor.h`
- Create: `D:\w\AIEngine\aicore\src\postprocess\nms_processor.cpp`
- Create: `D:\w\AIEngine\aicore\tests\test_preprocess.cpp`

- [ ] **Step 1: 实现 ResizeProcessor**

```cpp
#pragma once
#include "core/processor.h"

namespace aicore {

class ResizeProcessor : public IProcessor {
public:
    std::string name() const override { return "resize"; }
    Status init(const Config& config) override {
        width_ = config.value("width", 640);
        height_ = config.value("height", 640);
        return Status{};
    }
    Status process(const Frame& input, Frame& output) override {
        cv::resize(input.image, output.image, cv::Size(width_, height_));
        output.width = width_;
        output.height = height_;
        output.id = input.id;
        return Status{};
    }
    Status destroy() override { return Status{}; }
private:
    int width_ = 640, height_ = 640;
};

} // namespace aicore
```

- [ ] **Step 2: 实现 NormalizeProcessor**

```cpp
#pragma once
#include "core/processor.h"

namespace aicore {

class NormalizeProcessor : public IProcessor {
public:
    std::string name() const override { return "normalize"; }
    Status init(const Config& config) override {
        auto meanArr = config["mean"];
        auto stdArr = config["std"];
        for (size_t i = 0; i < 3 && i < meanArr.size(); ++i) {
            mean_[i] = meanArr[i].get<double>();
            std_[i] = stdArr[i].get<double>();
        }
        return Status{};
    }
    Status process(const Frame& input, Frame& output) override {
        input.image.convertTo(output.image, CV_32FC3, 1.0 / 255.0);
        // 逐通道归一化
        std::vector<cv::Mat> channels(3);
        cv::split(output.image, channels);
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean_[i]) / std_[i];
        }
        cv::merge(channels, output.image);
        output.id = input.id;
        output.width = input.width;
        output.height = input.height;
        return Status{};
    }
    Status destroy() override { return Status{}; }
private:
    double mean_[3] = {0.485, 0.456, 0.406};
    double std_[3] = {0.229, 0.224, 0.225};
};

} // namespace aicore
```

- [ ] **Step 3: 实现 NmsProcessor**

```cpp
#pragma once
#include "core/processor.h"
#include <algorithm>

namespace aicore {

class NmsProcessor : public IProcessor {
public:
    std::string name() const override { return "nms"; }
    Status init(const Config& config) override {
        iouThreshold_ = config.value("iou_threshold", 0.5);
        confThreshold_ = config.value("conf_threshold", 0.3);
        return Status{};
    }
    Status process(const Frame& input, Frame& output) override {
        std::vector<NodeResult> detections;
        // 收集所有检测结果
        for (auto& [nodeId, result] : input.nodeResults) {
            if (result.confidence >= confThreshold_) {
                detections.push_back(result);
            }
        }

        // 按置信度排序
        std::sort(detections.begin(), detections.end(),
            [](const NodeResult& a, const NodeResult& b) {
                return a.confidence > b.confidence;
            });

        // 简化 NMS：只保留置信度最高的，过滤 IOU 超过阈值的
        std::vector<bool> keep(detections.size(), true);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (iou(detections[i].bbox, detections[j].bbox) > iouThreshold_) {
                    keep[j] = false;
                }
            }
        }

        output = input;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (keep[i]) {
                output.nodeResults["nms_" + detections[i].nodeId] = detections[i];
            }
        }
        return Status{};
    }
    Status destroy() override { return Status{}; }

private:
    float iouThreshold_ = 0.5f;
    float confThreshold_ = 0.3f;

    static float iou(const BBox& a, const BBox& b) {
        float interX = std::max(0.0f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
        float interY = std::max(0.0f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
        float interArea = interX * interY;
        float unionArea = a.w * a.h + b.w * b.h - interArea;
        return unionArea > 0 ? interArea / unionArea : 0;
    }
};

} // namespace aicore
```

- [ ] **Step 4: 测试预处理节点**

```cpp
#include <gtest/gtest.h>
#include "preprocess/resize_processor.h"
#include "preprocess/normalize_processor.h"
#include "postprocess/nms_processor.h"

using namespace aicore;

TEST(PreprocessTest, Resize) {
    ResizeProcessor rp;
    Config cfg = {{"width", 224}, {"height", 224}};
    ASSERT_TRUE(rp.init(cfg));

    Frame in, out;
    in.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    ASSERT_TRUE(rp.process(in, out));
    EXPECT_EQ(out.image.rows, 224);
    EXPECT_EQ(out.image.cols, 224);
}

TEST(PreprocessTest, Normalize) {
    NormalizeProcessor np;
    Config cfg = {
        {"mean", {0.485, 0.456, 0.406}},
        {"std", {0.229, 0.224, 0.225}}
    };
    ASSERT_TRUE(np.init(cfg));

    Frame in, out;
    in.image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    ASSERT_TRUE(np.process(in, out));
    EXPECT_EQ(out.image.type(), CV_32FC3);
}

TEST(PostprocessTest, NmsFilter) {
    NmsProcessor nms;
    Config cfg = {{"iou_threshold", 0.5}, {"conf_threshold", 0.3}};
    ASSERT_TRUE(nms.init(cfg));

    Frame in;
    NodeResult r1; r1.nodeId = "a"; r1.confidence = 0.9f; r1.bbox = {0, 0, 100, 100};
    NodeResult r2; r2.nodeId = "b"; r2.confidence = 0.4f; r2.bbox = {10, 10, 90, 90};  // 高 IOU，应被过滤
    NodeResult r3; r3.nodeId = "c"; r3.confidence = 0.2f; r3.bbox = {200, 200, 50, 50}; // 低置信度，应被过滤
    in.nodeResults["a"] = r1;
    in.nodeResults["b"] = r2;
    in.nodeResults["c"] = r3;

    Frame out;
    ASSERT_TRUE(nms.process(in, out));
    // 只保留 a（最高置信度，b 因高 IOU 被过滤，c 因低置信度被过滤）
    EXPECT_EQ(out.nodeResults.size(), 1);
    EXPECT_GT(out.nodeResults.count("nms_a"), 0);
}
```

---

### Task 10: DLL C API 导出层

**Files:**
- Create: `D:\w\AIEngine\aicore\include\api\aicore_api.h`
- Create: `D:\w\AIEngine\aicore\src\api\aicore_api.cpp`
- Create: `D:\w\AIEngine\aicore\samples\qt_integration\main.cpp`

- [ ] **Step 1: 创建 aicore_api.h**

```cpp
#pragma once
#include "core/types.h"

extern "C" {

// 创建引擎实例，返回不透明句柄
// configPath: JSON 配置文件路径
// 返回引擎句柄，失败返回 nullptr
AICORE_API void* AICore_Create(const char* configPath);

// 执行推理
// handle: 引擎句柄
// imageData: 图像数据指针 (RGB, 连续)
// width, height, channels: 图像尺寸
// outJson: 输出 JSON 缓冲区
// outBufSize: 缓冲区大小
// 返回 0 成功，非 0 错误码
AICORE_API int AICore_Run(void* handle,
                          const unsigned char* imageData,
                          int width, int height, int channels,
                          char* outJson, int outBufSize);

// 销毁引擎实例
AICORE_API void AICore_Destroy(void* handle);

} // extern "C"
```

- [ ] **Step 2: 创建 aicore_api.cpp**

```cpp
#include "api/aicore_api.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"
#include <opencv2/core.hpp>
#include <sstream>

namespace aicore {
namespace {

class EngineHolder {
public:
    std::unique_ptr<IPipeline> pipeline;
    PipelineConfig config;
};

EngineHolder* toHolder(void* handle) {
    return static_cast<EngineHolder*>(handle);
}

} // anonymous namespace
} // namespace aicore

using namespace aicore;

AICORE_API void* AICore_Create(const char* configPath) {
    try {
        auto holder = std::make_unique<EngineHolder>();
        holder->config = ConfigParser::parseFile(configPath);

        auto& builder = PipelineBuilder::globalInstance();
        holder->pipeline = builder.build(holder->config);
        if (!holder->pipeline) {
            return nullptr;
        }

        return holder.release();
    } catch (const std::exception& e) {
        // 日志输出已超出本函数职责
        return nullptr;
    }
}

AICORE_API int AICore_Run(void* handle,
                          const unsigned char* imageData,
                          int width, int height, int channels,
                          char* outJson, int outBufSize) {
    if (!handle || !imageData || !outJson) return -1;

    try {
        auto holder = toHolder(handle);

        // 从裸数据构造 cv::Mat（零拷贝）
        cv::Mat image(height, width, channels == 1 ? CV_8UC1 : CV_8UC3,
                      const_cast<unsigned char*>(imageData));

        Frame input;
        input.image = image.clone();  // 拷贝，因为输入数据可能在调用后释放
        input.width = width;
        input.height = height;
        input.channels = channels;

        auto status = holder->pipeline->run(input);
        if (status.code != StatusCode::OK) {
            return static_cast<int>(status.code);
        }

        // 序列化为 JSON
        Config resultJson;
        resultJson["timestamp"] = input.id;
        resultJson["status"] = static_cast<int>(status.code);

        for (auto& [nodeId, result] : input.nodeResults) {
            Config item;
            item["node_id"] = result.nodeId;
            item["label"] = result.label;
            item["confidence"] = result.confidence;
            item["bbox"] = {{"x", result.bbox.x},
                            {"y", result.bbox.y},
                            {"w", result.bbox.w},
                            {"h", result.bbox.h}};
            resultJson["detections"].push_back(item);
        }

        std::string jsonStr = resultJson.dump();
        if (jsonStr.size() >= static_cast<size_t>(outBufSize)) {
            return -2; // 缓冲区不足
        }
        memcpy(outJson, jsonStr.c_str(), jsonStr.size() + 1);

        return 0;
    } catch (const std::exception& e) {
        return -3;
    }
}

AICORE_API void AICore_Destroy(void* handle) {
    delete toHolder(handle);
}
```

- [ ] **Step 3: 创建 Qt 集成示例**

```cpp
// samples/qt_integration/main.cpp
// Qt 上位机集成示例
#include <QCoreApplication>
#include <QDebug>
#include <QLibrary>
#include <opencv2/opencv.hpp>

// 函数指针类型定义
typedef void* (*CreateFn)(const char*);
typedef int (*RunFn)(void*, const unsigned char*, int, int, int, char*, int);
typedef void (*DestroyFn)(void*);

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);

    // 加载 DLL
    QLibrary lib("aicore.dll");
    if (!lib.load()) {
        qCritical() << "Failed to load aicore.dll:" << lib.errorString();
        return -1;
    }

    auto createFn = (CreateFn)lib.resolve("AICore_Create");
    auto runFn = (RunFn)lib.resolve("AICore_Run");
    auto destroyFn = (DestroyFn)lib.resolve("AICore_Destroy");

    if (!createFn || !runFn || !destroyFn) {
        qCritical() << "Failed to resolve API functions";
        return -1;
    }

    // 创建引擎
    void* engine = createFn("config.json");
    if (!engine) {
        qCritical() << "Failed to create engine";
        return -1;
    }

    // 读取图像并推理
    cv::Mat image = cv::imread("test.jpg");
    if (image.empty()) {
        qWarning() << "No test image found";
    } else {
        char buffer[65536];
        int ret = runFn(engine, image.data, image.cols, image.rows, image.channels(),
                        buffer, sizeof(buffer));
        if (ret == 0) {
            qDebug() << "Result:" << buffer;
        } else {
            qWarning() << "Inference failed with code:" << ret;
        }
    }

    // 销毁
    destroyFn(engine);
    return 0;
}
```

---

### Task 11: 完整更新 CMakeLists.txt + 编译验证

**Files:**
- Modify: `D:\w\AIEngine\aicore\CMakeLists.txt`
- Modify: `D:\w\AIEngine\aicore\tests\CMakeLists.txt`

- [ ] **Step 1: 更新 CMakeLists.txt，添加所有源文件**

```cmake
cmake_minimum_required(VERSION 3.20)
project(aicore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.7.0 REQUIRED)

set(AICORE_SOURCES
    src/api/aicore_api.cpp
    src/config/config_parser.cpp
    src/config/pipeline_builder.cpp
    src/engine/thread_pool.cpp
    src/engine/engine_pool.cpp
    src/pipeline/pipeline_impl.cpp
    src/pipeline/model_node.cpp
    src/pipeline/composite_node.cpp
    src/pipeline/merge_node.cpp
    src/backend/backend_factory.cpp
    src/backend/tensorrt_backend.cpp
    src/backend/onnxruntime_backend.cpp
    src/backend/libtorch_backend.cpp
)

set(AICORE_HEADERS
    include/core/types.h
    include/core/frame.h
    include/core/processor.h
    include/core/model_backend.h
    include/core/pipeline.h
    include/core/allocator.h
    include/pipeline/pipeline_impl.h
    include/pipeline/model_node.h
    include/pipeline/composite_node.h
    include/pipeline/merge_node.h
    include/backend/backend_factory.h
    include/backend/tensorrt_backend.h
    include/backend/onnxruntime_backend.h
    include/backend/libtorch_backend.h
    include/preprocess/resize_processor.h
    include/preprocess/normalize_processor.h
    include/postprocess/nms_processor.h
    include/config/config_parser.h
    include/config/pipeline_builder.h
    include/engine/thread_pool.h
    include/engine/engine_pool.h
    include/api/aicore_api.h
)

set(NLOHMANN_JSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party)

add_library(aicore SHARED ${AICORE_SOURCES} ${AICORE_HEADERS})
target_include_directories(aicore PUBLIC include)
target_include_directories(aicore PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
target_include_directories(aicore PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aicore PRIVATE ${OpenCV_LIBS})
target_compile_definitions(aicore PRIVATE AICORE_EXPORTS)

# 默认构建 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 2: 更新 tests/CMakeLists.txt**

```cmake
find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_types.cpp
    test_processor.cpp
    test_config_parser.cpp
    test_pipeline_builder.cpp
    test_pipeline.cpp
    test_composite_node.cpp
    test_merge_node.cpp
    test_thread_pool.cpp
    test_preprocess.cpp
    test_backend_factory.cpp
)

add_executable(aicore_tests ${TEST_SOURCES})
target_include_directories(aicore_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(aicore_tests PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
target_include_directories(aicore_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aicore_tests PRIVATE GTest::GTest GTest::Main ${OpenCV_LIBS})
add_test(NAME aicore_tests COMMAND aicore_tests)
```

- [ ] **Step 3: 完整编译并运行全部测试**

Run: `cd D:\w\AIEngine\aicore\build && cmake .. && cmake --build . --config Release`
Run: `ctest --output-on-failure`
Expected: All tests pass

- [ ] **Step 4: 实现后端空桩**

创建 TensorRTBackend / ONNXRuntimeBackend / LibTorchBackend 的空实现，返回 ErrorInternal，确保编译通过。

```cpp
// src/backend/tensorrt_backend.cpp
#include "backend/tensorrt_backend.h"
namespace aicore {
TensorRTBackend::TensorRTBackend() = default;
TensorRTBackend::~TensorRTBackend() = default;
std::string TensorRTBackend::name() const { return "tensorrt"; }
Status TensorRTBackend::init(const Config&) { return Status{StatusCode::ErrorInternal, "Not implemented yet"}; }
Status TensorRTBackend::process(const Frame&, Frame&) { return Status{StatusCode::ErrorInternal, "Not implemented yet"}; }
Status TensorRTBackend::infer(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status{StatusCode::ErrorInternal, "Not implemented yet"}; }
ModelInfo TensorRTBackend::modelInfo() const { return {}; }
Status TensorRTBackend::destroy() { return Status{}; }
}
```

---

## 计划审核

计划完成后，将进行 plan-document-reviewer 审核。

## 执行交接

计划保存到 `docs/superpowers/plans/2026-06-10-aicore-phase1-inference-engine.md` 后，将提供两种执行选项。
