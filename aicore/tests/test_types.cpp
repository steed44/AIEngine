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
