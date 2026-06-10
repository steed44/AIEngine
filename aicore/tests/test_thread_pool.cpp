#include <gtest/gtest.h>
#include "engine/thread_pool.h"
#include "engine/engine_pool.h"
#include <atomic>

using namespace aicore;

TEST(ThreadPoolTest, BasicEnqueue) {
    ThreadPool pool(2);
    auto future = pool.Enqueue([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPoolTest, ParallelExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    constexpr int kTasks = 100;

    std::vector<std::future<void>> futures;
    for (int i = 0; i < kTasks; ++i) {
        futures.push_back(pool.Enqueue([&counter] { counter.fetch_add(1); }));
    }
    for (auto& f : futures) f.get();

    EXPECT_EQ(counter.load(), kTasks);
}

TEST(ThreadPoolTest, WaitAll) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};
    constexpr int kTasks = 10;

    for (int i = 0; i < kTasks; ++i) {
        pool.Enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }
    pool.WaitAll();
    EXPECT_EQ(counter.load(), kTasks);
}

TEST(ThreadPoolTest, PendingCount) {
    ThreadPool pool(1);
    pool.Enqueue([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
    pool.Enqueue([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
    EXPECT_GE(pool.GetPendingCount(), 0);
    pool.WaitAll();
    EXPECT_EQ(pool.GetPendingCount(), 0);
}

TEST(EnginePoolTest, AcquireRelease) {
    EnginePool pool(2);
    ModelInfo info;
    info.backend = BackendType::kTensorRT;
    info.modelPath = "dummy";

    std::shared_ptr<IModelBackend> engine;
    Status s = pool.Acquire("model_a", info, engine);
    EXPECT_TRUE(s);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->GetBackendType(), BackendType::kTensorRT);

    s = pool.Release("model_a", engine);
    EXPECT_TRUE(s);
}

TEST(EnginePoolTest, ReuseEngine) {
    EnginePool pool(2);
    ModelInfo info;
    info.backend = BackendType::kTensorRT;
    info.modelPath = "dummy";

    std::shared_ptr<IModelBackend> engine1;
    ASSERT_TRUE(pool.Acquire("model_a", info, engine1));
    ASSERT_TRUE(pool.Release("model_a", engine1));

    std::shared_ptr<IModelBackend> engine2;
    ASSERT_TRUE(pool.Acquire("model_a", info, engine2));
    EXPECT_EQ(engine1.get(), engine2.get());
}

TEST(EnginePoolTest, Clear) {
    EnginePool pool(2);
    ModelInfo info;
    info.backend = BackendType::kTensorRT;
    info.modelPath = "dummy";

    std::shared_ptr<IModelBackend> engine;
    ASSERT_TRUE(pool.Acquire("model_a", info, engine));
    ASSERT_TRUE(pool.Release("model_a", engine));
    pool.Clear();

    std::shared_ptr<IModelBackend> engine2;
    ASSERT_TRUE(pool.Acquire("model_a", info, engine2));
    EXPECT_NE(engine2, nullptr);
}
