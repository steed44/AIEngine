# Graph Report - .  (2026-06-24)

## Corpus Check
- 184 files · ~280,937 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 684 nodes · 1273 edges · 27 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 393 edges (avg confidence: 0.51)
- Token cost: 18,000 input · 0 output

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
10. `aicore.dll 推理核心DLL` - 16 edges

## Surprising Connections (you probably didn't know these)
- `TrainingDialog()` --calls--> `buildUi()`  [INFERRED]
  aicore\gui\training_dialog.h → aicore\gui\training_dialog.cpp
- `aicore.dll 推理核心DLL` --references--> `Pipeline DAG编排执行器`  [EXTRACTED]
  aicore/CMakeLists.txt → docs/superpowers/specs/2026-06-10-aicore-engine-design.md
- `aicore.dll 推理核心DLL` --references--> `C API 外部接口层`  [EXTRACTED]
  aicore/CMakeLists.txt → README.md
- `模型优化层 (Model Optimization Layer)` --conceptually_related_to--> `Phase 2: 模型优化工具 (PyTorch->ONNX->TensorRT)`  [INFERRED]
  基于NVIDIA GPU工控机的工业检测纯C++ AI引擎技术方案.md → docs/superpowers/plans/2026-06-10-aicore-phase2-optimizer.md
- `graphify 知识图谱工具` --references--> `AIEngine`  [EXTRACTED]
  AGENTS.md → README.md

## Hyperedges (group relationships)
- **训练-优化-推理 三级架构** — training_layer, model_optimization_layer, inference_layer [EXTRACTED 1.00]
- **PatchCore 多ROI异常检测系统** — patchcore_node, multi_roi_node, memory_bank, coreset_sampler, patchcore_trainer, ibackbone, folder_dataset, roi_trainer [EXTRACTED 1.00]
- **PyTorch->ONNX->TensorRT 模型优化流程** — python_embedding, onnx_exporter, tensorrt_builder, int8_calibrator, model_optimizer, pytorch_onnx_tensorrt_path [EXTRACTED 1.00]

## Communities

### Community 0 - "JSON Library"
Cohesion: 0.05
Nodes (110): accept(), add(), array(), at(), back(), basic_json(), begin(), binary() (+102 more)

### Community 1 - "Engine Core Runtime"
Cohesion: 0.04
Nodes (15): ~AiEngine(), Shutdown(), Parse(), ParseBackend(), Iou(), Process(), AddEdge(), AddNode() (+7 more)

### Community 2 - "Training Module"
Cohesion: 0.04
Nodes (12): aicore_trainer.dll 训练模块DLL, DataLoader(), HasNext(), Next(), Reset(), Get(), Size(), SimpleCNN (+4 more)

### Community 3 - "Inference Engine"
Cohesion: 0.04
Nodes (28): aicore.dll 推理核心DLL, AICoreUI.exe Qt5上位机, 异步执行与流水线并行 (3-stage pipeline), C API 外部接口层, L2Dist(), Sample(), Faster R-CNN 双阶段检测模型, Scheduler GPU显存调度器 (+20 more)

### Community 4 - "C API Layer"
Cohesion: 0.07
Nodes (18): aicore_engine_execute(), aicore_engine_init(), aicore_pipeline_create(), aicore_pipeline_execute(), aicore_result_to_json(), StoreError(), drawAnomalyOverlay(), drawDetections() (+10 more)

### Community 5 - "Model Optimization"
Cohesion: 0.07
Nodes (11): aicore_optimizer.dll 模型优化工具DLL, 推理层 (Inference Layer), 模型优化层 (Model Optimization Layer), Phase 2: 模型优化工具 (PyTorch->ONNX->TensorRT), ensureInitialized(), Finalize(), Initialize(), PythonEmbedding() (+3 more)

### Community 6 - "Backend Factory & Stubs"
Cohesion: 0.07
Nodes (6): LibTorchBackendStub, ONNXRuntimeBackendStub, StubBase, TensorRTBackendStub, 多流推理+引擎池 优化策略, ModelRegistryTest

### Community 7 - "CUDA Memory Management"
Cohesion: 0.1
Nodes (17): EvictLRU(), MemoryManager(), QueryTotalVRAM(), TryAlloc(), CreateTestBankFile(), CreateZerosBankFile(), TEST(), Clear() (+9 more)

### Community 8 - "Qt UI & Training Dialog"
Cohesion: 0.15
Nodes (19): YOLOTrainerTest, appendLog(), browseDir(), browseFile(), buildPatchCoreTab(), buildUi(), buildYoloTab(), onProgressTick() (+11 more)

### Community 9 - "Inference Server & Batch"
Cohesion: 0.11
Nodes (10): DynamicBatcher 动态批处理, BatcherLoop(), ExecuteBatch(), InferAsync(), ~InferenceServer(), InferSync(), LoadModel(), ReplaceModel() (+2 more)

### Community 10 - "Backbone Implementation"
Cohesion: 0.15
Nodes (4): Init(), SplitLayerNames(), Init(), SplitLayerNames()

### Community 11 - "YOLO Data Pipeline"
Cohesion: 0.17
Nodes (13): YOLODataTest, collate(), Get(), hsvJitter(), letterbox(), Load(), loadBatch(), loadLabels() (+5 more)

### Community 12 - "YOLO Model Definition"
Cohesion: 0.13
Nodes (2): forward(), predict()

### Community 13 - "YOLO Loss Functions"
Cohesion: 0.26
Nodes (6): assignTargets(), bceLoss(), ciouLoss(), computeIoU(), dflLoss(), operator()()

### Community 14 - "Core Architecture Interfaces"
Cohesion: 0.22
Nodes (10): aicore_api.h (C API 入口), CreateBackbone (工厂函数: opencv_dnn/model_backend/libtorch), IBackbone (patchcore/ 纯虚接口: Init/Extract/GetType), IProcessor (core/ 纯虚接口: Init / Process), MemoryBank (特征存储+NN搜索), MultiRoiNode (多ROI共享IBackbone), PipelineBuilder (config/ 静态Build(json)), PipelineImpl (IPipeline 实现, DAG 编排) (+2 more)

### Community 15 - "PatchCore Backbone Python"
Cohesion: 0.28
Nodes (5): main(), PatchCoreBackboneWrapper, 包装模型，截取指定中间层的输出作为特征元组返回      工作原理：         在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列, 创建 forward hook，将中间层输出追加到 _features 列表, 前向传播，返回指定中间层的特征图元组          参数:             x: 输入张量，形状 [N, C, H, W]，值范围 [0, 1]

### Community 16 - "ONNX Export"
Cohesion: 0.4
Nodes (4): export_onnx(), 将 TorchScript 模型导出为 ONNX 格式      参数:         config_json: JSON 字符串，包含 model_path, 使用 Ultralytics YOLO 训练模型      参数:         config_json: JSON 字符串，包含训练参数     返回:, train_yolo()

### Community 17 - "YOLO Training Python"
Cohesion: 0.67
Nodes (2): 训练 YOLO 模型      参数:         cfg_json: JSON 字符串，包含 data/epochs/imgsz/batch 等训练参数, train()

### Community 18 - "DGN Sample"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "GPU Stubs"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Sanity Test"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Project Meta"
Cohesion: 1.0
Nodes (2): AIEngine, graphify 知识图谱工具

### Community 22 - "IModelBackend"
Cohesion: 1.0
Nodes (1): IModelBackend 推理后端抽象

### Community 23 - "IPipeline"
Cohesion: 1.0
Nodes (1): IPipeline 流水线编排器

### Community 24 - "CheckpointManager"
Cohesion: 1.0
Nodes (1): CheckpointManager 检查点管理器

### Community 25 - "Zero-Copy"
Cohesion: 1.0
Nodes (1): 零拷贝数据传输 (Zero-Copy)

### Community 26 - "Architecture Diagram"
Cohesion: 1.0
Nodes (1): AIEngine Architecture Diagram (DOT rendered)

## Knowledge Gaps
- **41 isolated node(s):** `将 TorchScript 模型导出为 ONNX 格式      参数:         config_json: JSON 字符串，包含 model_path`, `使用 Ultralytics YOLO 训练模型      参数:         config_json: JSON 字符串，包含训练参数     返回:`, `包装模型，截取指定中间层的输出作为特征元组返回      工作原理：         在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列`, `创建 forward hook，将中间层输出追加到 _features 列表`, `前向传播，返回指定中间层的特征图元组          参数:             x: 输入张量，形状 [N, C, H, W]，值范围 [0, 1]` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `DGN Sample`** (2 nodes): `dgn.cpp`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GPU Stubs`** (2 nodes): `stubs_gpu.cpp`, `BatchL2DistanceGPU()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sanity Test`** (2 nodes): `test_sanity.cpp`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Meta`** (2 nodes): `AIEngine`, `graphify 知识图谱工具`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `IModelBackend`** (1 nodes): `IModelBackend 推理后端抽象`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `IPipeline`** (1 nodes): `IPipeline 流水线编排器`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CheckpointManager`** (1 nodes): `CheckpointManager 检查点管理器`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Zero-Copy`** (1 nodes): `零拷贝数据传输 (Zero-Copy)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Architecture Diagram`** (1 nodes): `AIEngine Architecture Diagram (DOT rendered)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `aicore.dll 推理核心DLL` connect `Inference Engine` to `Inference Server & Batch`, `Backend Factory & Stubs`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
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
  _41 weakly-connected nodes found - possible documentation gaps or missing edges._