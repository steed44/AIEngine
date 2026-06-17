// ============================================================
// 文件: tests/test_trainer.cpp
// 用途: 训练模块全套单元测试
//   涵盖 Dataset / DataLoader / Augmentation / ModelFactory /
//   TrainingLoop / Scheduler / Checkpoint / EarlyStopping /
//   Validator / Exporter / Callback / API
// ============================================================

#include <gtest/gtest.h>
#include "trainer/data/dataset.h"
#include "trainer/data/data_loader.h"
#include "trainer/data/augmentation.h"
#include "trainer/model/model_factory.h"
#include "trainer/model/python_trainer.h"
#include "trainer/training/training_loop.h"
#include "trainer/training/training_scheduler.h"
#include "trainer/training/checkpoint.h"
#include "trainer/training/early_stopping.h"
#include "trainer/validation/validator.h"
#include "trainer/export/model_exporter.h"
#include "trainer/callback.h"
#include "trainer/trainer_api.h"

using namespace aicore;

// 测试：创建 COCO 数据集
TEST(DatasetTest, CreateCOCO) {
    auto ds = std::make_shared<COCODataset>();
    EXPECT_TRUE(ds->Load("data/coco.json"));
    // 由于 COCODataset 非公开类型，使用类型擦除句柄方式
    // 仅验证接口工作正常
    EXPECT_EQ(ds->NumClasses(), 80);
}

// 测试：DataLoader 批次计算和迭代
TEST(DataLoaderTest, BasicIteration) {
    class TestDataset : public IDataset {
    public:
        Status Load(const std::string&) override { return Status{}; }
        size_t Size() const override { return 10; }
        Sample Get(size_t) override { return {cv::Mat(10, 10, CV_8UC3), 0, "", {}}; }
        int NumClasses() const override { return 2; }
    };

    auto ds = std::make_shared<TestDataset>();
    DataLoader loader(ds, 4, false);
    EXPECT_EQ(loader.NumBatches(), 3);  // 10 个样本，batch=4 → 3 批次

    auto batch = loader.Next();
    EXPECT_EQ(batch.images.size(), 4);
    EXPECT_EQ(batch.labels.size(), 4);
}

// 测试：空增强流水线直接通过输入
TEST(AugmentationPipelineTest, EmptyPipeline) {
    AugmentationPipeline pipe;
    Sample input{cv::Mat(10, 10, CV_8UC3), 1, "", {}};
    auto out = pipe.Apply(input);
    EXPECT_EQ(out.label, 1);
}

// 测试：ModelFactory 创建简单 CNN 模型
TEST(ModelFactoryTest, CreateSimpleCNN) {
    auto model = ModelFactory::Create(ModelArch::kSimpleCNN);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->GetArchName(), "simple_cnn");

    ModelConfig cfg;
    EXPECT_TRUE(model->Build(cfg));
    EXPECT_TRUE(model->Save("test_model.pt"));
}

// 测试：PythonTrainer 创建成功，错误信息为空
TEST(PythonTrainerTest, Create) {
    PythonTrainer trainer;
    EXPECT_EQ(trainer.GetLastError(), "");
}

// 测试：训练调度器添加任务后 PendingCount 正确
TEST(TrainingSchedulerTest, AddAndRun) {
    TrainingScheduler scheduler;
    TrainTask task;
    task.modelId = "yolov8n";
    task.configPath = "config.json";

    EXPECT_TRUE(scheduler.AddTask(task));
    EXPECT_EQ(scheduler.PendingCount(), 1);
    EXPECT_TRUE(scheduler.RunAll());
    EXPECT_EQ(scheduler.PendingCount(), 0);
}

// 测试：CheckpointManager 保存和清理
TEST(CheckpointManagerTest, Create) {
    CheckpointManager cm("checkpoints");
    std::string data = "dummy";
    EXPECT_TRUE(cm.Save(1, 0.5f, data));
    EXPECT_TRUE(cm.Cleanup(3));
}

// 测试：EarlyStopping 在指标持续改善时不停止
TEST(EarlyStoppingTest, NoStopOnImprovement) {
    EarlyStopping es(3, 0.01f);
    EXPECT_FALSE(es.ShouldStop(0.5f));
    EXPECT_FALSE(es.ShouldStop(0.6f));
    EXPECT_FALSE(es.ShouldStop(0.7f));
}

// 测试：EarlyStopping 超过耐心轮数后停止
TEST(EarlyStoppingTest, StopAfterPatience) {
    EarlyStopping es(2, 0.01f);
    EXPECT_FALSE(es.ShouldStop(0.5f));
    EXPECT_FALSE(es.ShouldStop(0.45f));
    EXPECT_FALSE(es.ShouldStop(0.44f));
    EXPECT_TRUE(es.ShouldStop(0.43f));
}

// 测试：EarlyStopping Reset 后重新计数
TEST(EarlyStoppingTest, Reset) {
    EarlyStopping es(2, 0.01f);
    es.ShouldStop(1.0f);
    es.ShouldStop(0.9f);
    es.ShouldStop(0.8f);
    EXPECT_TRUE(es.ShouldStop(0.7f));
    es.Reset();
    EXPECT_FALSE(es.ShouldStop(0.5f));
}

// 测试：Validator 空验证返回成功
TEST(ValidatorTest, ValidateEmpty) {
    Validator validator;
    ValidationResult result;
    auto s = validator.Validate(nullptr, nullptr, result);
    EXPECT_TRUE(s);
}

// 测试：ModelExporter 导出 ONNX
TEST(ModelExporterTest, Create) {
    ModelExporter exporter;
    auto s = exporter.ExportToONNX("nonexistent.pt", "out.onnx");
    EXPECT_TRUE(s);
}

// 测试用回调实现 — 记录各阶段调用次数
class TestCallback : public ITrainCallback {
public:
    void OnEpochBegin(int epoch) override { beginCalls++; lastEpoch = epoch; }
    void OnEpochEnd(int epoch, float loss, float metric) override { endCalls++; }
    void OnBatchEnd(int batch, float loss) override { batchCalls++; }
    void OnTrainEnd(float bestMetric) override { trainEndCalls++; }
    int beginCalls = 0, endCalls = 0, batchCalls = 0, trainEndCalls = 0, lastEpoch = -1;
};

// 测试：TrainingLoop 在训练过程中正确调用回调
TEST(TrainingLoopTest, RunCallsCallbacks) {
    auto cb = std::make_shared<TestCallback>();
    TrainingLoop loop;
    loop.AddCallback(cb);

    TrainConfig cfg;
    cfg.epochs = 3;
    auto ds = std::make_shared<COCODataset>();
    AugmentationPipeline aug;

    EXPECT_TRUE(loop.Run(cfg, ds, ds, aug));
    EXPECT_EQ(cb->endCalls, 3);       // 3 个 epoch
    EXPECT_EQ(cb->trainEndCalls, 1);  // 1 次训练结束
}

// 测试：训练 API 版本号非空
TEST(TrainerApiTest, Version) {
    auto ver = aicore_trainer_version();
    EXPECT_NE(ver, nullptr);
    EXPECT_NE(std::string(ver), "");
}

// 测试：无效配置导致训练 API 返回非零
TEST(TrainerApiTest, TrainInvalidConfig) {
    const char* err = nullptr;
    int ret = aicore_train_run("invalid json", &err);
    EXPECT_NE(ret, 0);
}
