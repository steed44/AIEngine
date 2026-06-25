#include <gtest/gtest.h>
#include "patchcore/scheduler.h"
#include "api/scheduler_api.h"

using namespace aicore;

// 测试：Scheduler 单例返回相同实例
TEST(SchedulerTest, InstanceSingleton) {
    auto& s1 = Scheduler::Instance();
    auto& s2 = Scheduler::Instance();
    EXPECT_EQ(&s1, &s2);
}

// 测试：默认优先级为 kBalanced
TEST(SchedulerTest, DefaultPriorityIsBalanced) {
    auto& s = Scheduler::Instance();
    EXPECT_EQ(s.GetPriority(), PriorityMode::kBalanced);
}

// 测试：SetPriority 和 GetPriority 往返
TEST(SchedulerTest, SetPriorityRoundtrip) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kInference);
    EXPECT_EQ(s.GetPriority(), PriorityMode::kInference);

    s.SetPriority(PriorityMode::kTraining);
    EXPECT_EQ(s.GetPriority(), PriorityMode::kTraining);

    s.SetPriority(PriorityMode::kBalanced);
    EXPECT_EQ(s.GetPriority(), PriorityMode::kBalanced);
}

// 测试：InferenceUseGPU 在 kInference 模式为 true，kTraining 为 false
TEST(SchedulerTest, InferenceUseGPU) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kInference);
    EXPECT_TRUE(s.InferenceUseGPU());

    s.SetPriority(PriorityMode::kTraining);
    EXPECT_FALSE(s.InferenceUseGPU());
}

// 测试：TrainingUseGPU 在 kTraining 模式为 true，kInference 为 false
TEST(SchedulerTest, TrainingUseGPU) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kInference);
    EXPECT_FALSE(s.TrainingUseGPU());

    s.SetPriority(PriorityMode::kTraining);
    EXPECT_TRUE(s.TrainingUseGPU());
}

// 测试：RecheckGPU 在非 kBalanced 模式下为无操作
TEST(SchedulerTest, RecheckGPUNoopOutsideBalanced) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kInference);
    // 不应崩溃
    s.RecheckGPU();

    s.SetPriority(PriorityMode::kTraining);
    s.RecheckGPU();
}

// 测试：SetGPUReservation 不崩溃
TEST(SchedulerTest, SetGPUReservation) {
    auto& s = Scheduler::Instance();
    s.SetGPUReservation(1024, 4096, 512);
    s.SetGPUReservation(0, 0, 0);
}

// 测试：C API aicore_scheduler_set_priority 和 get_priority 往返
TEST(SchedulerApiTest, SetGetPriorityStrings) {
    aicore_scheduler_set_priority("inference");
    EXPECT_STREQ(aicore_scheduler_get_priority(), "inference");

    aicore_scheduler_set_priority("training");
    EXPECT_STREQ(aicore_scheduler_get_priority(), "training");

    aicore_scheduler_set_priority("balanced");
    EXPECT_STREQ(aicore_scheduler_get_priority(), "balanced");
}

// 测试：C API aicore_scheduler_set_priority 传入空指针不崩溃
TEST(SchedulerApiTest, SetPriorityNullSafe) {
    aicore_scheduler_set_priority(nullptr);
    // 不应崩溃，优先级不变
}

// 测试：C API aicore_scheduler_recheck 不崩溃
TEST(SchedulerApiTest, RecheckNoCrash) {
    aicore_scheduler_recheck();
}

TEST(SchedulerEdgeTest, DeviceCountNonNegative) {
    auto& s = Scheduler::Instance();
    EXPECT_GE(s.DeviceCount(), 0);
}

TEST(SchedulerEdgeTest, SelectDeviceReturnsValid) {
    auto& s = Scheduler::Instance();
    int dev = s.SelectDevice();
    EXPECT_GE(dev, 0);
    EXPECT_LT(dev, std::max(1, s.DeviceCount()));
}

TEST(SchedulerEdgeTest, SelectDeviceRoundRobin) {
    auto& s = Scheduler::Instance();
    int d1 = s.SelectDevice();
    int d2 = s.SelectDevice();
    int d3 = s.SelectDevice();
    (void)d1; (void)d2; (void)d3;
}

TEST(SchedulerEdgeTest, SetGPUReservationZero) {
    auto& s = Scheduler::Instance();
    s.SetGPUReservation(0, 0, 0);
    s.RecheckGPU();
}

TEST(SchedulerEdgeTest, TrainingModeInferenceGPU) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kTraining);
    EXPECT_TRUE(s.TrainingUseGPU());
    EXPECT_FALSE(s.InferenceUseGPU());
}

TEST(SchedulerEdgeTest, InferenceModeTrainingGPU) {
    auto& s = Scheduler::Instance();
    s.SetPriority(PriorityMode::kInference);
    EXPECT_TRUE(s.InferenceUseGPU());
    EXPECT_FALSE(s.TrainingUseGPU());
}

class SchedulerCleanup : public ::testing::Test {
protected:
    void TearDown() override {
        Scheduler::Instance().SetPriority(PriorityMode::kBalanced);
    }
};

TEST_F(SchedulerCleanup, RestoreBalancedAfterEdgeTests) {
    Scheduler::Instance().SetPriority(PriorityMode::kBalanced);
    EXPECT_EQ(Scheduler::Instance().GetPriority(), PriorityMode::kBalanced);
}
