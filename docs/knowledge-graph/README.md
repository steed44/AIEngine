# AIEngine 代码架构知识图谱

## 文件说明

| 文件 | 格式 | 用途 |
|------|------|------|
| `aicore_architecture.dot` | Graphviz DOT | 类继承/模块依赖/数据流可视化图 |
| `knowledge_graph.json` | JSON-LD | 机器可读的知识图谱（类、接口、工厂、数据流、设计决策） |

## 渲染 DOT 图

### 方式 1: Graphviz 命令行

```bash
# 安装 Graphviz (https://graphviz.org/download/)
dot -Tpng -Gdpi=150 docs/knowledge-graph/aicore_architecture.dot -o docs/knowledge-graph/aicore_architecture.png
```

### 方式 2: VS Code

安装 **Graphviz (dot) language support** 插件，直接预览 `.dot` 文件。

### 方式 3: 在线渲染

将 DOT 文件内容粘贴到 [https://dreampuf.github.io/GraphvizOnline/](https://dreampuf.github.io/GraphvizOnline/)。

## 知识图谱内容

### 模块依赖图（顶层）

```
api (C API)
  └─ pipeline (DAG 编排)
       ├─ core (基础类型/接口)
       ├─ config (配置解析/管线构建)
       ├─ engine (线程池/模型池)
       ├─ preprocess (Resize/Normalize)
       ├─ postprocess (NMS)
       └─ patchcore (异常检测)
            ├─ backbone (IBackbone + 3 实现)
            ├─ memory_bank (PatchFeature 存储 + NN 搜索)
            ├─ coreset (最远点采样)
            ├─ patchcore_node (单 ROI 推理)
            ├─ multi_roi_node (多 ROI 推理)
            ├─ patchcore_trainer (单 ROI 训练)
            └─ roi_trainer (多 ROI 训练)
cli──┴─ optimizer* / trainer* (* = 桩实现)
```

### 类继承树

```
IProcessor (core/)
  ├─ ResizeNode (preprocess/)
  ├─ NormalizeNode (preprocess/)
  ├─ NmsNode (postprocess/)
  ├─ ModelNode (pipeline/)
  ├─ MergeNode (pipeline/)
  ├─ CompositeNode (pipeline/)
  ├─ PatchCoreNode (patchcore/)
  └─ MultiRoiNode (patchcore/)

IBackbone (patchcore/)
  ├─ OpenCVDnnBackbone          → cv::dnn::Net (CPU)
  ├─ ModelBackendBackbone       → IModelBackend (通用)
  └─ LibTorchBackbone           → torch::jit::Module (GPU, 条件编译)

IDataset (trainer/)
  ├─ FolderDataset (patchcore/) → 专为 PatchCore 设计
  └─ COCODataset (trainer/)     → COCO 格式

IPipeline (core/)
  └─ PipelineImpl (pipeline/)   → DAG 实现
```

### 工厂函数

| 工厂 | 输入 | 输出 | 位置 |
|------|------|------|------|
| `CreateBackbone` | `type` (string) + `NodeConfig` | `unique_ptr<IBackbone>` | `patchcore/backbone.cpp` |
| `BackendFactory::Create` | `BackendType` enum | `unique_ptr<IModelBackend>` | `backend/backend_factory.cpp` |
| `ModelFactory::Create` | `ModelArch` enum | `unique_ptr<IModel>` | `trainer/model/model_factory.cpp` |
| `PipelineBuilder::Build` | `PipelineConfig` | `shared_ptr<IPipeline>` | `config/pipeline_builder.cpp` |

### 训练数据流

```
config_rois.json → MultiRoiConfig::FromJson()
                       │
                       ▼
                RoiTrainer::TrainAll()
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
      自动检测大图?           --no-stream
      是 → 流式                ▼
             │           TrainAllBatch()
             ▼           FolderDataset::Load()
  TrainAllStreaming()     全部加载到内存
  列出文件路径                     │
      │                   对每个 ROI:
      ▼                    crop → Trainer.Train()
  初始化 backbone × 1                  │
      │                    IDataset → Extract
  对每张图:                     → Coreset
    imread                     → MemoryBank
    for 每个 ROI:               → Save(.bin)
      cropper
      backbone.Extract
      append → roiFeatures[ri]
      │
      ▼
  对每个 ROI:
    truncate → Coreset → MemoryBank → Save(.bin)
```

### 推理数据流

```
config_rois.json → MultiRoiNode::Init()
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     CreateBackbone()     对每个 ROI:
     backbone.Init()      MemoryBank::Load(.bin)
            │             
            ▼             
     MultiRoiNode::Process(Frame)
            │
            ▼
     for 每个 RoiModelSlot:
       crop ROI
       backbone.Extract(crop) → PatchFeature[]
       slot.memoryBank.ComputeAnomalyMap() → heatmap + score
       overlay(rect, score, heatmap)
            │
            ▼
     output Frame (含所有 ROI 的检测结果)
```
