#include <gtest/gtest.h>
#include "server/model_registry.h"
#include "backend/backend_factory.h"

using namespace aicore;

class ModelRegistryTest : public ::testing::Test {
protected:
    ModelRegistry registry_;

    std::unique_ptr<IModelBackend> MakeStub() {
        return BackendFactory::Create(BackendType::kONNXRuntime);
    }
};

// 测试：GetActive 对未知模型返回空指针
TEST_F(ModelRegistryTest, GetActiveUnknownReturnsNull) {
    auto slot = registry_.GetActive("nonexistent");
    EXPECT_EQ(slot, nullptr);
}

// 测试：Replace 为不存在的模型创建新记录
TEST_F(ModelRegistryTest, ReplaceCreatesNewSlot) {
    auto backend = MakeStub();
    auto s = registry_.Replace("model_a", std::move(backend), 1024, 1);
    EXPECT_TRUE(s);

    auto slot = registry_.GetActive("model_a");
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->modelName, "model_a");
    EXPECT_EQ(slot->version, 1);
    EXPECT_EQ(slot->vramMB, 1024);
    // refCount 已被 GetActive 递增
    EXPECT_GE(slot->refCount.load(), 1);
}

// 测试：GetActive 递增 refCount
TEST_F(ModelRegistryTest, GetActiveIncrementsRefCount) {
    auto backend = MakeStub();
    registry_.Replace("model_b", std::move(backend), 512, 1);

    auto slot1 = registry_.GetActive("model_b");
    ASSERT_NE(slot1, nullptr);
    int rc1 = slot1->refCount.load();

    auto slot2 = registry_.GetActive("model_b");
    ASSERT_NE(slot2, nullptr);
    EXPECT_EQ(slot2->refCount.load(), rc1 + 1);
}

// 测试：Unload 移除模型
TEST_F(ModelRegistryTest, UnloadRemovesSlot) {
    auto backend = MakeStub();
    registry_.Replace("model_c", std::move(backend), 256, 1);
    EXPECT_NE(registry_.GetActive("model_c"), nullptr);

    registry_.Unload("model_c");
    EXPECT_EQ(registry_.GetActive("model_c"), nullptr);
}

// 测试：Unload 不存在的模型不崩溃
TEST_F(ModelRegistryTest, UnloadNonExistentNoCrash) {
    registry_.Unload("nonexistent");
}

// 测试：List 返回 JSON 格式字符串
TEST_F(ModelRegistryTest, ListReturnsJson) {
    auto backend = MakeStub();
    registry_.Replace("model_x", std::move(backend), 2048, 2);

    std::string json = registry_.List();
    EXPECT_NE(json.find("\"model_x\""), std::string::npos);
    EXPECT_NE(json.find("\"version\":2"), std::string::npos);
    EXPECT_NE(json.find("\"vramMB\":2048"), std::string::npos);
}

// 测试：空注册表 List 返回空数组
TEST_F(ModelRegistryTest, ListEmptyReturnsEmptyArray) {
    EXPECT_EQ(registry_.List(), "[]");
}

// 测试：EvictLRU 驱逐最早使用的空闲模型
TEST_F(ModelRegistryTest, EvictLRURemovesOldest) {
    for (int i = 0; i < 3; i++) {
        auto backend = MakeStub();
        std::string name = "model_" + std::to_string(i);
        registry_.Replace(name, std::move(backend), 100, 1);
    }
    // 模拟使用顺序：model_0 最早，model_2 最新
    // GetActive 递增 refCount，完成后手动释放
    for (int i = 0; i < 3; i++) {
        std::string name = "model_" + std::to_string(i);
        auto slot = registry_.GetActive(name);
        ASSERT_NE(slot, nullptr);
        slot->refCount.fetch_sub(1);  // 释放引用
    }

    // 需要 150MB → 应驱逐最早的空闲模型
    // 每个 100MB，需驱逐前 2 个共 200MB 才能满足
    auto s = registry_.EvictLRU(150);
    EXPECT_TRUE(s);

    // model_0 和 model_1 已被驱逐
    EXPECT_EQ(registry_.GetActive("model_0"), nullptr);
    EXPECT_EQ(registry_.GetActive("model_1"), nullptr);

    // model_2 最新，仍在
    EXPECT_NE(registry_.GetActive("model_2"), nullptr);
}

// 测试：EvictLRU 驱逐量不足时返回错误
TEST_F(ModelRegistryTest, EvictLRUInsufficientFreedReturnsError) {
    auto backend = MakeStub();
    registry_.Replace("small_model", std::move(backend), 100, 1);

    auto s = registry_.EvictLRU(999);
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorResourceExhaust);
}

// 测试：EvictLRU 正在使用的模型不会被驱逐
TEST_F(ModelRegistryTest, EvictLRUSkipsActiveModels) {
    auto backend1 = MakeStub();
    registry_.Replace("active_model", std::move(backend1), 500, 1);
    auto backend2 = MakeStub();
    registry_.Replace("idle_model", std::move(backend2), 500, 1);

    // 持有 active_model 的引用（refCount > 0）
    auto activeSlot = registry_.GetActive("active_model");

    auto s = registry_.EvictLRU(500);
    EXPECT_TRUE(s);

    // active_model 还在（refCount > 0），idle_model 被驱逐
    EXPECT_NE(registry_.GetActive("active_model"), nullptr);
    EXPECT_EQ(registry_.GetActive("idle_model"), nullptr);
}
