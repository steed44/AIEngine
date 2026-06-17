// ============================================================
// 文件: tests/test_processor.cpp
// 用途: Frame 数据结构和 IProcessor 接口的单元测试
// ============================================================

#include <gtest/gtest.h>
#include "core/frame.h"
#include "core/processor.h"
#include "core/pipeline.h"
#include "core/allocator.h"

using namespace aicore;

// 测试：Frame 默认构造为空
TEST(FrameTest, DefaultConstructor) {
    Frame f;
    EXPECT_TRUE(f.empty());
    EXPECT_EQ(f.frameId, 0);
}

// 测试：用 cv::Mat 构造 Frame，尺寸正确
TEST(FrameTest, MatConstructor) {
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    Frame f(img, 1);
    EXPECT_FALSE(f.empty());
    EXPECT_EQ(f.frameId, 1);
    EXPECT_EQ(f.width(), 640);
    EXPECT_EQ(f.height(), 480);
}

// 测试：Frame 移动语义 — 原 Mat 被移动后为空
TEST(FrameTest, MoveSemantics) {
    cv::Mat img(100, 200, CV_8UC3);
    Frame f(std::move(img), 42);
    EXPECT_FALSE(f.empty());
    EXPECT_EQ(f.width(), 200);
    EXPECT_TRUE(img.empty());
}

// 测试用 IProcessor 实现 — 简单复制输入到输出
class TestProcessor : public IProcessor {
public:
    Status Init(const NodeConfig& config) override {
        config_ = config;
        return Status{};
    }
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override {
        for (const auto& in : inputs) {
            outputs.push_back(Frame(in.image.clone(), in.frameId));
        }
        return Status{};
    }
    std::string GetName() const override { return "test_processor"; }
    std::string GetType() const override { return "test"; }
    NodeConfig config_;
};

// 测试：IProcessor 基本流程 — Init / GetName / GetType
TEST(IProcessorTest, BasicFlow) {
    TestProcessor proc;
    NodeConfig cfg{{"param1", "value1"}};
    EXPECT_TRUE(proc.Init(cfg));
    EXPECT_EQ(proc.GetName(), "test_processor");
    EXPECT_EQ(proc.GetType(), "test");
    EXPECT_EQ(proc.config_["param1"], "value1");
}

// 测试：Process 方法复制输入图像（输出数据指针不同）
TEST(IProcessorTest, ProcessCopiesInput) {
    TestProcessor proc;
    proc.Init({});

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
    Frame input(img, 1);
    std::vector<Frame> inputs = {input};
    std::vector<Frame> outputs;

    EXPECT_TRUE(proc.Process(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_NE(outputs[0].image.data, img.data);
    EXPECT_EQ(outputs[0].frameId, 1);
}

// 测试：PipelineState 枚举值互不相同
TEST(PipelineStateTest, EnumValues) {
    EXPECT_NE(static_cast<int>(PipelineState::kCreated),
              static_cast<int>(PipelineState::kReady));
    EXPECT_NE(static_cast<int>(PipelineState::kRunning),
              static_cast<int>(PipelineState::kStopped));
}
