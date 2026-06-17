// ============================================================
// 文件: tests/test_thread_pool.cpp
// 用途: 线程池 (ThreadPool) 和引擎池 (EnginePool) 单元测试
// ============================================================

#include <gtest/gtest.h>
#include "engine/thread_pool.h"
#include "engine/engine_pool.h"
#include <atomic>

using namespace aicore;

// 测试：线程池基本入队和执行
TEST(ThreadPoolTest, BasicEnqueue) {
    ThreadPool pool(2);
    auto future = pool.Enqueue([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// 测试：线程池并行执行 100 个任务
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

// 测试：线程池 WaitAll 等待所有任务完成
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

// 测试：线程池待处理任务计数
TEST(ThreadPoolTest, PendingCount) {
    ThreadPool pool(1);
    pool.Enqueue([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
    pool.Enqueue([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
    EXPECT_GE(pool.GetPendingCount(), 0);
    pool.WaitAll();
    EXPECT_EQ(pool.GetPendingCount(), 0);
}

// 测试：引擎池获取/释放引擎
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

// 测试：引擎池复用已释放的引擎（同一指针）
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

// 测试：引擎池 Clear 后仍然可以获取新引擎
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
