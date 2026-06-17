// ============================================================
// 文件: tests/test_backend_factory.cpp
// 用途: 后端工厂 (BackendFactory) 创建各类型引擎的单元测试
// ============================================================

#include <gtest/gtest.h>
#include "core/model_backend.h"
#include "backend/backend_factory.h"

using namespace aicore;

// 测试：创建 TensorRT 后端返回非空指针且类型正确
TEST(BackendFactoryTest, CreateTensorRT) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kTensorRT);
    EXPECT_FALSE(backend->IsLoaded());
}

// 测试：创建 ONNX Runtime 后端返回非空指针
TEST(BackendFactoryTest, CreateONNXRuntime) {
    auto backend = BackendFactory::Create(BackendType::kONNXRuntime);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kONNXRuntime);
}

// 测试：创建 LibTorch 后端返回非空指针
TEST(BackendFactoryTest, CreateLibTorch) {
    auto backend = BackendFactory::Create(BackendType::kLibTorch);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kLibTorch);
}

// 测试：创建未知类型后端返回空指针
TEST(BackendFactoryTest, CreateUnknownReturnsNull) {
    auto backend = BackendFactory::Create(BackendType::kUnknown);
    EXPECT_EQ(backend, nullptr);
}

// 测试：ModelInfo 默认构造函数填充默认值
TEST(ModelInfoTest, DefaultConstructor) {
    ModelInfo info;
    EXPECT_EQ(info.batchSize, 1);
    EXPECT_EQ(info.deviceId, 0);
    EXPECT_EQ(info.backend, BackendType::kUnknown);
    EXPECT_TRUE(info.modelPath.empty());
}

// 测试：设置和获取设备 ID
TEST(ModelBackendTest, SetDeviceId) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    backend->SetDeviceId(1);
    EXPECT_EQ(backend->GetDeviceId(), 1);
}
