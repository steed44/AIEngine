// ============================================================
// 文件: tests/test_types.cpp
// 用途: 核心类型 (Status / Tensor / Result) 默认值和基本行为测试
// ============================================================

#include <gtest/gtest.h>
#include "core/types.h"

using namespace aicore;

// 测试：Status 默认构造为 OK
TEST(TypesTest, StatusOk) {
    Status s;
    EXPECT_TRUE(s);
    EXPECT_EQ(s.code, StatusCode::OK);
}

// 测试：Status 可以构造为带错误码和消息的错误状态
TEST(TypesTest, StatusError) {
    Status s{StatusCode::ErrorInternal, "test error"};
    EXPECT_FALSE(s);
    EXPECT_EQ(s.message, "test error");
}

// 测试：Tensor 默认构造值为 float32 / CPU / nullptr
TEST(TypesTest, TensorDefaults) {
    Tensor t;
    EXPECT_EQ(t.dtype, DataType::kFloat32);
    EXPECT_EQ(t.memory, MemoryType::kCPU);
    EXPECT_EQ(t.data, nullptr);
}

// 测试：Result 默认构造时间戳为 0，检测列表为空，状态为 OK
TEST(TypesTest, ResultDefaults) {
    Result r;
    EXPECT_EQ(r.timestamp, 0);
    EXPECT_TRUE(r.detections.empty());
    EXPECT_EQ(r.status, StatusCode::OK);
}
