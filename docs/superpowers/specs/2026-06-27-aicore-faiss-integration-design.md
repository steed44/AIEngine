# FAISS 近似最近邻搜索集成设计文档

## 目标

在现有 PatchCore MemoryBank 暴力最近邻搜索基础上，集成 FAISS 的 IVF / HNSW 近似最近邻搜索算法，使 PatchCore 推理能在百万级特征库上高效运行。

## 功能范围

- FAISS 索引构建：训练阶段从 MemoryBank 构建 IVF/HNSW/Flat 索引
- FAISS 索引推理：三种算法可选（暴力/IVF/HNSW），配置参数控制
- 自动降级：FAISS 索引损坏或构建失败时回退暴力搜索
- GPU 版 FAISS（可选）：CMake 选项 `FAISS_ENABLE_GPU=ON` 启用 GPU 索引
- 现有流程零破坏：MemoryBank / TieredMemoryBank 代码不变

## 不包含

- FAISS 索引增量更新（不支持在线学习）
- 多 GPU FAISS 分布式搜索
- IndexIVFPQ / IndexScalarQuantizer 等压缩索引
- Python FAISS 绑定（仅 C++ 集成）

## 架构

```
PatchCoreNode (IProcessor)
    │
    ├── TieredMemoryBank                ← 现有，不变
    │       ├── ComputeAnomalyMap()     ← 暴力搜索路径
    │       └── PromoteToGPU()          ← GPU CUDA kernel 路径
    │
    └── FaissIndexBridge (新增)         ← FAISS 搜索路径
            ├── Ti: IndexFlatL2         ← 精确搜索（FAISS 优化版）
            ├── IVF: IndexIVFFlat       ← 倒排索引，O(log N) 近似搜索
            └── HNSW: IndexHNSWFlat     ← 图索引，O(log N) 近似搜索

训练流程（不变）：
    IDataset → Backbone.Extract → CoresetSampler → MemoryBank.Save(.bin)
    ↓ 新增（可选）
    └── FaissIndexBridge.TrainFromMemoryBank(.bin) → FaissIndex.Save(.hnsw.faiss)
```

### 数据流（推理）

```
输入 Frame
    │
    ├── [暴力模式] Backbone.Extract → TieredMemoryBank.ComputeAnomalyMap
    │
    └── [FAISS 模式] Backbone.Extract → FaissIndexBridge.ComputeAnomalyMap
                                            │
                                     FaissIndex.SearchBatch()
                                            │
                                    得分图 → 上采样 → 异常热力图
```

## 文件结构

```
aicore/
├── include/patchcore/
│   ├── faiss_index.h                 ← 新增：FaissIndex、FaissIndexConfig、枚举
│   └── faiss_index_bridge.h          ← 新增：FaissIndexBridge（桥接层）
├── src/patchcore/
│   ├── faiss_index.cpp               ← 新增：IndexFlat/IVF/HNSW 封装
│   └── faiss_index_bridge.cpp        ← 新增：Bridge 实现、ComputeAnomalyMap
├── tests/
│   └── test_faiss_index.cpp          ← 新增：单元测试
└── CMakeLists.txt                    ← 修改：FetchContent FAISS
```

## 设计细节

### FaissIndexConfig 和 FaissSearchAlgorithm

```cpp
// search_algorithm 枚举
enum class FaissSearchAlgorithm {
    BruteForce = 0,  // 等价 IndexFlatL2，精确搜索
    IVF,             // IndexIVFFlat
    HNSW             // IndexHNSWFlat
};

// FAISS 索引配置参数
struct FaissIndexConfig {
    FaissSearchAlgorithm algorithm = FaissSearchAlgorithm::BruteForce;
    // 特征向量维度（必填，Train() 前必须设置）
    int d = 0;

    // --- IVF 参数 ---
    int nlist = 100;    // IVF 聚类中心数（构建参数）
    int nprobe = 16;    // IVF 搜索时探查簇数（精度→速度权衡）

    // --- HNSW 参数 ---
    int M = 16;             // HNSW 每层最大连接数
    int efConstruction = 200;  // HNSW 构建时动态列表大小
    int efSearch = 64;     // HNSW 推理时动态列表大小

    // --- 通用 ---
    int gpuDevice = -1;    // -1=CPU, >=0=GPU 设备号
    uint32_t seed = 42;    // 随机种子（IVF k-means 可重现）
};
```

### FaissIndex 类

```cpp
class FaissIndex {
public:
    ~FaissIndex();

    // 从 PatchFeature 列表训练并构建 FAISS 索引
    // 流程：
    //   1. 检查特征非空、维度一致
    //   2. 根据 algorithm 创建对应 index
    //      BruteForce → IndexFlatL2（无需 train）
    //      IVF        → IndexIVFFlat，需要 train + add
    //      HNSW       → IndexHNSWFlat（无需 train，只需 add）
    //   3. 设置搜索参数（nprobe / efSearch）
    Status Train(const std::vector<PatchFeature>& features,
                 const FaissIndexConfig& cfg);

    // 序列化保存 FAISS 索引到文件
    // 格式：FAISS 内置 write_index() 二进制格式
    Status Save(const std::string& path) const;

    // 从文件加载 FAISS 索引
    Status Load(const std::string& path);

    // k=1 最近邻搜索
    // 返回 {bestIdx, bestDist}
    // O(1) for Flat, O(nlist*nprobe) for IVF, O(log N) for HNSW
    std::pair<size_t, float> NearestNeighbor(
        const std::vector<float>& query) const;

    // 批量搜索
    // 输入 queries: M × d 浮点矩阵
    // 输出 distances: M 个 float（最近邻距离）
    // 批量搜索利用 FAISS 内部 SIMD/BLAS 优化
    std::vector<float> SearchBatch(
        const std::vector<float>& queries, int numQueries) const;

    // 访问器
    size_t Size() const { return ntotal_; }
    int Dimension() const { return d_; }
    FaissSearchAlgorithm Algorithm() const { return cfg_.algorithm; }
    bool IsTrained() const { return is_trained_; }

private:
    std::unique_ptr<faiss::Index> index_;  // FAISS 索引实例（多态）
    FaissIndexConfig cfg_;
    bool is_trained_ = false;
    size_t ntotal_ = 0;
    int d_ = 0;
};
```

### FaissIndexBridge 类

桥接 MemoryBank（`.bin` 文件）和 FaissIndex 的适配层。

```cpp
class FaissIndexBridge {
public:
    // 从 MemoryBank .bin 文件路径读取特征数据并训练 FAISS 索引
    //   - 读取 .bin 文件解析特征
    //   - 调用 FaissIndex.Train() 构建索引
    //   - 同时保存索引到 .{algorithm}.faiss 文件（下次加速加载）
    //   - 失败时返回错误状态
    Status TrainFromMemoryBank(const std::string& memoryBankPath,
                                const FaissIndexConfig& cfg);

    // 直接加载已经训练好的 .faiss 索引文件
    //   - 不依赖 .bin 文件
    //   - 启动最快（无需重新训练）
    Status LoadIndex(const std::string& indexPath);

    // 将当前索引保存到文件
    Status SaveIndex(const std::string& indexPath) const;

    // 计算异常热力图（与 TieredMemoryBank::ComputeAnomalyMap 接口兼容）
    //   - 从 PatchFeature 列表提取特征矩阵
    //   - 调用 FaissIndex.SearchBatch 批量搜索
    //   - 将结果距离填入得分图 → 上采样到原图尺寸
    //   - 多层融合：max-fusion
    std::vector<float> ComputeAnomalyMap(
        const std::vector<PatchFeature>& queries,
        int imgH, int imgW) const;

    const FaissIndex& GetIndex() const { return index_; }
    bool IsReady() const { return index_.IsTrained(); }

private:
    FaissIndex index_;
    std::string binPath_;  // 原始 .bin 文件路径
};
```

### PatchCoreNode 集成

PatchCoreNode 新增可选 FAISS 搜索路径，通过配置 `search_algorithm` 控制。

```cpp
class PatchCoreNode : public IProcessor {
private:
    // ... 现有成员 ...
    TieredMemoryBank memoryBank_;      // 已有，暴力搜索
    
    // 新增 FAISS 相关
    std::unique_ptr<FaissIndexBridge> faissBridge_;  // FAISS 桥接（可选）
    FaissSearchAlgorithm searchAlgo_ = FaissSearchAlgorithm::BruteForce;
    
    // FAISS 参数（从 JSON 配置读取）
    int faissNlist_ = 100;
    int faissNprobe_ = 16;
    int faissM_ = 16;
    int faissEfConstruction_ = 200;
    int faissEfSearch_ = 64;
};
```

**Init 流程扩展：**

```
PatchCoreNode::Init(config)
    1. 读取 search_algorithm（可选，默认 "brute_force"）
    2. 读取 FAISS 参数（nlist/nprobe/M/efConstruction/efSearch）
    3. 加载 memory bank .bin（现有逻辑不变）
    4. 如果 search_algorithm != "brute_force"：
        a. 尝试加载 .{algo}.faiss 索引文件
        b. 如果文件不存在，从 memory bank 训练
        c. 如果训练失败，打警告日志，降级到暴力搜索
    5. 继续现有初始化：PromoteToGPU / backbone init / 线程池
```

**Process 流程扩展：**

```
PatchCoreNode::Process(inputs, outputs)
    1. 提取图像特征（Backbone.Extract，不变）
    2. 如果 searchAlgo_ == BruteForce：
        走 memoryBank_.ComputeAnomalyMap（现有逻辑）
    3. 否则（IVF / HNSW）：
        走 faissBridge_->ComputeAnomalyMap
    4. 后处理（取最大得分 / 阈值判断，不变）
```

### 文件格式命名约定

```
memory_bank.bin                  ← 现有格式，不变
memory_bank.ivf.faiss           ← IVF FAISS 索引
memory_bank.hnsw.faiss          ← HNSW FAISS 索引
memory_bank.flat.faiss          ← Flat FAISS 索引（可选）
```

加载优先级：`{memory_bank_path}.{algorithm}.faiss`。如果不存在，自动训练后保存。

JSON 配置也可显式指定 `faiss_index_path` 覆盖默认路径。

### 配置参数（JSON）

```json
{
    "pipeline": {
        "nodes": [
            {
                "id": "patchcore",
                "type": "patchcore",
                "params": {
                    "model_path": "wideresnet.onnx",
                    "memory_bank_path": "memory_bank.bin",
                    
                    "search_algorithm": "hnsw",
                    "faiss_nlist": 200,
                    "faiss_nprobe": 32,
                    "faiss_m": 24,
                    "faiss_ef_construction": 300,
                    "faiss_ef_search": 128
                }
            }
        ]
    }
}
```

可选值：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `search_algorithm` | `"brute_force"` | `"brute_force"` / `"ivf"` / `"hnsw"` |
| `faiss_nlist` | 100 | IVF 聚类中心数 |
| `faiss_nprobe` | 16 | IVF 搜索探查簇数 |
| `faiss_m` | 16 | HNSW 连接数 |
| `faiss_ef_construction` | 200 | HNSW 构建动态列表 |
| `faiss_ef_search` | 64 | HNSW 搜索动态列表 |

### 训练阶段扩展

PatchCoreTrainConfig 新增可选参数：

```cpp
struct PatchCoreTrainConfig {
    // ... 现有参数 ...

    // FAISS 索引构建（可选）
    bool buildFaissIndex = false;                    // 训练时是否构建 FAISS 索引
    FaissSearchAlgorithm faissAlgorithm = FaissSearchAlgorithm::HNSW;
    int faissNlist = 100;
    int faissNprobe = 16;
    int faissM = 16;
    int faissEfConstruction = 200;
    int faissEfSearch = 64;
};
```

训练结束前新增步骤：
```
PatchCoreTrainer::Train()
    // ... 现有流程：提取 → Coreset → Build → Save
    
    if (cfg.buildFaissIndex):
        1. 读取刚保存的 memory_bank.bin
        2. 构建 FaissIndex（IVF 或 HNSW）
        3. 保存到 memory_bank.{algorithm}.faiss
```

### 降级策略

```
启动降级链（Init）：
  search_algorithm == "brute_force"
    → 跳 FAISS，走暴力搜索路径 ✓
  
  search_algorithm == "ivf" / "hnsw"
    → 尝试 LoadIndex(.faiss)
      → 成功 ✓
      → 失败 → TrainFromMemoryBank(.bin)
        → 成功 ✓
        → 失败 → 日志警告，降级到 brute_force

运行时降级（Process）：
  faissBridge_ 未 ready
    → 走 memoryBank_.ComputeAnomalyMap ✓
  
  特征维度不匹配
    → 返回错误状态（不应静默）
```

## 测试计划

### 单元测试（test_faiss_index.cpp）

| 测试 | 验证点 |
|------|--------|
| FaissIndexFlat_TrainAndSearch | 3个特征训练，搜索应返回最近邻 |
| FaissIndexIVF_TrainAndSearch | IVF 训练后搜索，结果在容忍误差内 |
| FaissIndexHNSW_TrainAndSearch | HNSW 训练后搜索，结果同 |
| FaissIndex_SaveLoadRoundtrip | 索引序列化后加载，搜索结果一致 |
| FaissIndex_EmptyPool | 0 特征训练返回失败 |
| FaissIndex_DimMismatch | 不同维度特征返回失败 |
| FaissIndexBridge_TrainFromMemoryBank | 从 .bin 构建 FAISS 索引 |
| FaissIndexBridge_ComputeAnomalyMap | 异常热力图尺寸和值正确 |
| FaissIndexBridge_FallbackOnCorrupt | 损坏的 .faiss 降级到暴力搜索 |
| PatchCoreNode_SearchAlgorithmConfig | JSON 配置 search_algorithm 生效 |
| PatchCoreNode_InvalidAlgoFallsBack | 未知算法名降级且不打崩 |

### 集成测试

```
test_patchcore.cpp 扩展现有 IntegrationTest:
  1. 暴力搜索结果（baseline）→ 记录最近邻距离
  2. FAISS 近似搜索 → 记录最近邻距离
  3. 对比两者差异在 5% 以内（recall@1 ≥ 0.95）
```

## CMake 集成

```cmake
# 在顶层 CMakeLists.txt 或 aicore/CMakeLists.txt 中添加
include(FetchContent)

# FAISS CPU 版（默认，无 CUDA 依赖）
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  faiss
  GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
  GIT_TAG v1.10.0
)
FetchContent_MakeAvailable(faiss)

# 链接到 aicore 目标
target_link_libraries(aicore PRIVATE faiss)
target_include_directories(aicore PRIVATE ${faiss_SOURCE_DIR})
```

GPU 版 FAISS（可选）：设置 `FAISS_ENABLE_GPU=ON` 并安装 CUDA 工具包即可启用。`FaissIndexConfig::gpuDevice` 设为 ≥0 时使用 GPU 索引，-1 使用 CPU。
