# Graph Report - .  (2026-06-17)

## Corpus Check
- 142 files · ~228,797 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 548 nodes · 1020 edges · 27 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 330 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `namespace()` - 22 edges
2. `push_back()` - 22 edges
3. `end()` - 22 edges
4. `begin()` - 21 edges
5. `basic_json()` - 20 edges
6. `is_object()` - 19 edges
7. `handle_value()` - 17 edges
8. `size()` - 17 edges
9. `patch_inplace()` - 17 edges
10. `erase()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Multi-ROI 多区域多模型支持` --conceptually_related_to--> `PatchCoreNode (异常检测节点)`  [INFERRED]
  基于NVIDIA GPU工控机的工业检测纯C++ AI引擎技术方案.md → docs/superpowers/specs/2026-06-10-aicore-patchcore-design.md
- `Phase 2: 模型优化工具 (PyTorch→ONNX→TensorRT)` --references--> `模型优化层 (Model Optimization Layer)`  [EXTRACTED]
  docs/superpowers/plans/2026-06-10-aicore-phase2-optimizer.md → 基于NVIDIA GPU工控机的工业检测纯C++ AI引擎技术方案.md
- `ThreadPool (线程池)` --references--> `异步执行与流水线并行 (3-stage pipeline)`  [INFERRED]
  docs/superpowers/specs/2026-06-10-aicore-engine-design.md → 基于NVIDIA GPU工控机的工业检测纯C++ AI引擎技术方案.md
- `aicore_optimizer.dll 源文件组织` --references--> `ModelOptimizer (编排器)`  [EXTRACTED]
  aicore/CMakeLists.txt → docs/superpowers/specs/2026-06-10-aicore-phase2-optimizer-design.md
- `aicore_trainer.dll 源文件组织` --references--> `TrainingLoop (LibTorch训练循环)`  [EXTRACTED]
  aicore/CMakeLists.txt → docs/superpowers/specs/2026-06-10-aicore-phase3-trainer-design.md

## Communities

### Community 0 - "Pipeline 管线与节点"
Cohesion: 0.04
Nodes (13): Parse(), ParseBackend(), Iou(), Process(), AddEdge(), AddNode(), Build(), Execute() (+5 more)

### Community 1 - "训练模块核心"
Cohesion: 0.05
Nodes (8): DataLoader(), HasNext(), Next(), Reset(), Get(), Size(), SimpleCNN, TestCallback

### Community 2 - "架构设计与计划文档"
Cohesion: 0.04
Nodes (54): AIEngine Architecture Diagram (DOT rendered), aicore.dll 源文件组织, 工厂函数: CreateBackbone,BackendFactory,ModelFactory,PipelineBuilder, 推理数据流: config→MultiRoiNode→backbone→MemoryBank→heatmap, 类继承树: IProcessor→7个子类, IBackbone→3个子类, IDataset→2子类, 训练数据流: config→RoiTrainer→backbone→Coreset→MemoryBank→.bin, PatchCore实现计划 (MemoryBank+Coreset+Node+Trainer), Phase 1: 推理核心DLL (Pipeline+Strategy) (+46 more)

### Community 3 - "Backbone 特征提取"
Cohesion: 0.06
Nodes (14): Init(), SplitLayerNames(), Init(), SplitLayerNames(), L2Dist(), Sample(), ComputeAnomalyMap(), NearestNeighbor() (+6 more)

### Community 4 - "JSON 库 (nlohmann)"
Cohesion: 0.06
Nodes (15): clear(), crbegin(), crend(), from_json(), get_impl(), get_to(), items(), iterator_wrapper() (+7 more)

### Community 5 - "JSON 容器操作"
Cohesion: 0.19
Nodes (30): at(), begin(), cbegin(), cend(), contains(), convert(), count(), diff() (+22 more)

### Community 6 - "INT8 校准器"
Cohesion: 0.09
Nodes (0): 

### Community 7 - "后端引擎工厂"
Cohesion: 0.09
Nodes (4): LibTorchBackendStub, ONNXRuntimeBackendStub, StubBase, TensorRTBackendStub

### Community 8 - "Python 嵌入"
Cohesion: 0.1
Nodes (4): Finalize(), PythonEmbedding(), Clear(), RunAll()

### Community 9 - "JSON 序列化"
Cohesion: 0.2
Nodes (21): back(), binary(), boolean(), empty(), end_array(), end_object(), handle_value(), max_size() (+13 more)

### Community 10 - "优化器与训练器 DLL"
Cohesion: 0.1
Nodes (21): aicore_optimizer.dll 源文件组织, aicore_trainer.dll 源文件组织, FolderDataset (文件夹数据集), Int8Calibrator (INT8校准器), ModelOptimizer (编排器), OnnxExporter (ONNX导出编排器), PythonEmbedding (Python嵌入层), TensorRTBuilder (引擎构建器) (+13 more)

### Community 11 - "C 外部 API"
Cohesion: 0.15
Nodes (11): aicore_pipeline_create(), aicore_pipeline_execute(), aicore_result_to_json(), StoreError(), drawAnomalyOverlay(), drawDetections(), initPipeline(), MainWindow() (+3 more)

### Community 12 - "JSON 解析 (BSON/CBOR)"
Cohesion: 0.2
Nodes (16): accept(), array(), basic_json(), from_bjdata(), from_bson(), from_cbor(), from_msgpack(), from_ubjson() (+8 more)

### Community 13 - "JSON 类型系统"
Cohesion: 0.23
Nodes (15): erase(), get_impl_ptr(), get_ptr(), is_array(), is_binary(), is_boolean(), is_number(), is_number_float() (+7 more)

### Community 14 - "多 ROI 异常检测"
Cohesion: 0.19
Nodes (3): DrawRoiOverlay(), Process(), ProcessOneRoi()

### Community 15 - "JSON 解码"
Cohesion: 0.19
Nodes (13): add(), data(), decode(), dump_integer(), get_token_string(), hex_bytes(), is_negative_number(), json_sax_dom_callback_parser (+5 more)

### Community 16 - "PatchCore 模型导出脚本"
Cohesion: 0.28
Nodes (5): main(), PatchCoreBackboneWrapper, 包装模型，截取指定中间层的输出作为特征元组返回      工作原理：         在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列, 创建 forward hook，将中间层输出追加到 _features 列表, 前向传播，返回指定中间层的特征图元组          参数:             x: 输入张量，形状 [N, C, H, W]，值范围 [0, 1]

### Community 17 - "模型导出脚本"
Cohesion: 0.4
Nodes (4): export_onnx(), 将 TorchScript 模型导出为 ONNX 格式      参数:         config_json: JSON 字符串，包含 model_path, 使用 Ultralytics YOLO 训练模型      参数:         config_json: JSON 字符串，包含训练参数     返回:, train_yolo()

### Community 18 - "Qt 上位机"
Cohesion: 0.5
Nodes (4): AICoreUI.exe Qt5上位机, Phase 4: Qt上位机 (AICoreUI.exe), MainWindow (Qt上位机), Qt UI数据流 (QImage→C API→JSON→绘制)

### Community 19 - "YOLO 训练脚本"
Cohesion: 0.67
Nodes (2): 训练 YOLO 模型      参数:         cfg_json: JSON 字符串，包含 data/epochs/imgsz/batch 等训练参数, train()

### Community 20 - "冒烟测试"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "图谱报告元数据"
Cohesion: 1.0
Nodes (2): 28个社区: 社区4(C API),社区7(后端桩),社区8(Python嵌入),社区9(Qt UI),社区10(ROI), God Nodes: namespace/push_back/end/begin/basic_json 为核心抽象

### Community 22 - "LibTorch 后端"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "ONNX Runtime 后端"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "TensorRT 后端"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "CLI 可执行文件"
Cohesion: 1.0
Nodes (1): CLI可执行文件: ModelOptimizer/AICoreTrainer/PatchCoreTrain/RoiTrain/RoiInfer

### Community 26 - "测试源文件分组"
Cohesion: 1.0
Nodes (1): 测试源文件按Phase分组: Core→Optimizer→Trainer→PatchCore

## Knowledge Gaps
- **59 isolated node(s):** `将 TorchScript 模型导出为 ONNX 格式      参数:         config_json: JSON 字符串，包含 model_path`, `使用 Ultralytics YOLO 训练模型      参数:         config_json: JSON 字符串，包含训练参数     返回:`, `包装模型，截取指定中间层的输出作为特征元组返回      工作原理：         在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列`, `创建 forward hook，将中间层输出追加到 _features 列表`, `前向传播，返回指定中间层的特征图元组          参数:             x: 输入张量，形状 [N, C, H, W]，值范围 [0, 1]` (+54 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `冒烟测试`** (2 nodes): `test_sanity.cpp`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `图谱报告元数据`** (2 nodes): `28个社区: 社区4(C API),社区7(后端桩),社区8(Python嵌入),社区9(Qt UI),社区10(ROI)`, `God Nodes: namespace/push_back/end/begin/basic_json 为核心抽象`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LibTorch 后端`** (1 nodes): `libtorch_backend.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ONNX Runtime 后端`** (1 nodes): `onnxruntime_backend.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TensorRT 后端`** (1 nodes): `tensorrt_backend.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CLI 可执行文件`** (1 nodes): `CLI可执行文件: ModelOptimizer/AICoreTrainer/PatchCoreTrain/RoiTrain/RoiInfer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `测试源文件分组`** (1 nodes): `测试源文件按Phase分组: Core→Optimizer→Trainer→PatchCore`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `模型优化层 (Model Optimization Layer)` connect `架构设计与计划文档` to `优化器与训练器 DLL`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `namespace()` (e.g. with `to_json()` and `from_json()`) actually correct?**
  _`namespace()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `push_back()` (e.g. with `start_object()` and `start_array()`) actually correct?**
  _`push_back()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `end()` (e.g. with `namespace()` and `to_string()`) actually correct?**
  _`end()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `begin()` (e.g. with `namespace()` and `to_string()`) actually correct?**
  _`begin()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `basic_json()` (e.g. with `to_json()` and `set_parents()`) actually correct?**
  _`basic_json()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **What connects `将 TorchScript 模型导出为 ONNX 格式      参数:         config_json: JSON 字符串，包含 model_path`, `使用 Ultralytics YOLO 训练模型      参数:         config_json: JSON 字符串，包含训练参数     返回:`, `包装模型，截取指定中间层的输出作为特征元组返回      工作原理：         在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列` to the rest of the system?**
  _59 weakly-connected nodes found - possible documentation gaps or missing edges._