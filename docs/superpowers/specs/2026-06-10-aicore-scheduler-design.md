# GPU 显存调度器设计文档

## 目标

解决 PatchCore 训练和推理流水线并行时 GPU 显存竞争问题，提供运行时可切换的优先级策略，低优先级的 backbone 自动降级到 CPU。

## 架构

```
┌─────────────────────────────────────────────────┐
│                   Scheduler                      │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ PriorityMode │  │  GPU Memory Probe        │  │
│  │ (atomic)    │  │  cudaMemGetInfo → dev      │  │
│  └──────┬──────┘  └──────────┬───────────────┘  │
│         │                    │                  │
│         ▼                    ▼                  │
│  ┌────────────────────────────────────┐         │
│  │  InferenceUseGPU / TrainingUseGPU   │         │
│  └────────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
         │                  │
         ▼                  ▼
  PatchCoreNode       PatchCoreTrainer
  (推理)               (训练)
```

## 优先级模式

```cpp
enum class PriorityMode {
    kInference,  // 推理 GPU，训练 CPU
    kTraining,   // 训练 GPU，推理 CPU
    kBalanced    // 自动检测，尽量两者 GPU，不够退化推理优先
};
```

| 模式 | 推理 backbone | 训练 backbone |
|------|-------------|-------------|
| kInference | GPU | CPU |
| kTraining | CPU | GPU |
| kBalanced | GPU | GPU 或 CPU（按显存自动决策） |

## kBalanced 自动检测逻辑

```
ProbeBothFitOnGPU():
  1. cudaMemGetInfo → freeBytes
  2. reservedForInference  = inferMB_   // 默认 2048 MB，可通过 SetGPUReservation() 配置
  3. reservedForTraining   = trainMB_   // 默认 6144 MB
  4. reserveHeadroom       = headroomMB_ // 默认 1024 MB
  5. required = reservedForInference + reservedForTraining + reserveHeadroom
  6. if freeBytes >= required → 两者全走 GPU, balancedResult_ = true
  7. else → balancedResult_ = false, 退化为推理优先
```

阈值可通过 Scheduler 方法配置：
```cpp
void SetGPUReservation(int inferMB, int trainMB, int headroomMB);
```
```

## Scheduler API

```cpp
class Scheduler {
public:
    static Scheduler& Instance();

    void SetPriority(PriorityMode mode);
    PriorityMode GetPriority() const { return priorityMode_; }

    bool InferenceUseGPU() const;   // 推理 backbone 是否用 GPU
    bool TrainingUseGPU() const;    // 训练 backbone 是否用 GPU

    void RecheckGPU();              // 仅在 kBalanced 模式下重新探测显存
    void SetGPUReservation(int inferMB, int trainMB, int headroomMB);

private:
    Scheduler();
    void ProbeBothFitOnGPU();

    std::atomic<PriorityMode> priorityMode_{PriorityMode::kBalanced};
    std::atomic<bool> balancedResult_{false};
    std::atomic<int> inferMB_{2048}, trainMB_{6144}, headroomMB_{1024};
};

默认模式为 `kBalanced`：启动时自动探测显存，优先让训练和推理都跑在 GPU。
`SetGPUReservation()` 可在任意时刻调用，阈值是 `std::atomic<int>`，与 `RecheckGPU` 无竞争。

## 线程安全

- `priorityMode_` 为 `std::atomic<PriorityMode>`，无锁读取
- `balancedResult_` 为 `std::atomic<bool>`，kBalanced 模式下显存检测结果
- `SetPriority()` 写入原子变量即刻生效，**下一次** `Process()` / `Train()` 调用读取新值
- backbone 对象（`gpuBackbone_` / `cpuBackbone_`）在 Init() 后只读，`Process()` 在入口处按原子值选择其一，调用期间不切换。因此不存在 mid-Extract 切换问题

`InferenceUseGPU()` 逻辑（以下真值表前提为 GPU 设备可用，无 GPU 时所有模式均返回 false）：

| 模式 | 返回值 |
|------|--------|
| kInference | true |
| kTraining | false |
| kBalanced | true（始终 GPU） |

`TrainingUseGPU()` 逻辑（以下真值表前提为 GPU 设备可用，无 GPU 时所有模式均返回 false）：

| 模式 | 返回值 |
|------|--------|
| kInference | false |
| kTraining | true |
| kBalanced | balancedResult_（够显存→true，不够→false转CPU） |

即 kBalanced 退化为"推理保底 GPU，训练抢 GPU"。

## C API

```cpp
AICORE_C_API void aicore_scheduler_set_priority(const char* mode);
// mode: "inference" / "training" / "balanced"

AICORE_C_API const char* aicore_scheduler_get_priority();
```

## 文件结构

```
include/patchcore/scheduler.h     ← Scheduler 类
include/api/scheduler_api.h       ← C API 声明
src/patchcore/scheduler.cpp       ← Scheduler 实现
src/api/scheduler_api.cpp         ← C API 实现
```

## 接入改造

### PatchCoreNode

持有两个 backbone：
```cpp
std::unique_ptr<IBackbone> gpuBackbone_;  // LibTorchBackbone (GPU)
std::unique_ptr<IBackbone> cpuBackbone_;  // OpenCVDnnBackbone (CPU)
```

`Init()` 时创建两个 backbone，类型固定：
- `gpuBackbone_` = CreateBackbone("libtorch", config)
- `cpuBackbone_` = CreateBackbone("opencv_dnn", config)

`Process()` 时按 Scheduler 选择，GPU 推理也加 try-catch 兜底：
```cpp
if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
    try {
        auto patchFeatures = gpuBackbone_->Extract(img);
    } catch (const std::runtime_error& e) {
        // GPU OOM → 自动切 CPU 执行此帧
        auto patchFeatures = cpuBackbone_->Extract(img);
    }
} else {
    auto patchFeatures = cpuBackbone_->Extract(img);
}
```

### PatchCoreTrainer

`Train()` 开头查 Scheduler：
```cpp
if (Scheduler::Instance().TrainingUseGPU()) {
    backbone = CreateBackbone("libtorch", cfg);
} else {
    backbone = CreateBackbone("opencv_dnn", cfg);
}
```
训练循环中 catch OOM：

```cpp
Status PatchCoreTrainer::Train(...) {
    std::unique_ptr<IBackbone> backbone = CreateBackbone(
        Scheduler::Instance().TrainingUseGPU() ? "libtorch" : "opencv_dnn", cfg);

    for (size_t i = 0; i < dataset.Size(); ) {
        auto sample = dataset.Get(i);
        std::vector<PatchFeature> feats;
        try {
            feats = backbone->Extract(sample.image);
        } catch (const std::runtime_error& e) {
            Scheduler::Instance().RecheckGPU();
            if (Scheduler::Instance().GetPriority() == PriorityMode::kTraining) {
                // 训练优先模式下 GPU 仍 OOM → 硬错误，不静默降级
                return Status{StatusCode::ErrorGpuDevice,
                    "OOM in kTraining mode, cannot continue"};
            }
            if (!Scheduler::Instance().TrainingUseGPU()) {
                // kBalanced/kInference 降级成功：切换 CPU backbone 重试
                backbone = CreateBackbone("opencv_dnn", cfg);
                continue;  // 不移动 i，重试当前样本
            }
            // probe 未反映 OOM 但仍触发：强制切 CPU 重试
            backbone = CreateBackbone("opencv_dnn", cfg);
            continue;
        }
        allFeatures.insert(allFeatures.end(), feats.begin(), feats.end());
        i++;
    }
    // ...
}
```

## UI 集成

AICoreUI 菜单添加：

```
设置 → 推理优先
      训练优先
      均衡
```

选中后调用 `aicore_scheduler_set_priority`。

## 兼容性

- 无 LibTorch（无 AICORE_HAS_LIBTORCH）时，`gpuBackbone_` 为 nullptr
- 无 GPU 设备时，`InferenceUseGPU()` 和 `TrainingUseGPU()` 在所有模式下均返回 false，相当于始终走 CPU
- `kTraining` 模式下，训练用 GPU，推理走 CPU OpenCV dnn，需要 ONNX 模型

## 测试

- Scheduler：断言三种模式返回正确的 GPU/CPU 选择
- kBalanced：模拟 cudaMemGetInfo 返回值 → 验证退化逻辑
- PatchCoreNode：切换模式 → 验证 Process 使用正确的 backbone
- C API：`aicore_scheduler_set_priority("inference")` → 再读回验证
