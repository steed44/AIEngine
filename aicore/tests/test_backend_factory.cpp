#include <gtest/gtest.h>
#include "core/model_backend.h"
#include "backend/backend_factory.h"

using namespace aicore;

TEST(BackendFactoryTest, CreateTensorRT) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kTensorRT);
    EXPECT_FALSE(backend->IsLoaded());
}

TEST(BackendFactoryTest, CreateONNXRuntime) {
    auto backend = BackendFactory::Create(BackendType::kONNXRuntime);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kONNXRuntime);
}

TEST(BackendFactoryTest, CreateLibTorch) {
    auto backend = BackendFactory::Create(BackendType::kLibTorch);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetBackendType(), BackendType::kLibTorch);
}

TEST(BackendFactoryTest, CreateUnknownReturnsNull) {
    auto backend = BackendFactory::Create(BackendType::kUnknown);
    EXPECT_EQ(backend, nullptr);
}

TEST(ModelInfoTest, DefaultConstructor) {
    ModelInfo info;
    EXPECT_EQ(info.batchSize, 1);
    EXPECT_EQ(info.deviceId, 0);
    EXPECT_EQ(info.backend, BackendType::kUnknown);
    EXPECT_TRUE(info.modelPath.empty());
}

TEST(ModelBackendTest, SetDeviceId) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    backend->SetDeviceId(1);
    EXPECT_EQ(backend->GetDeviceId(), 1);
}
