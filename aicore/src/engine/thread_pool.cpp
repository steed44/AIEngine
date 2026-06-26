// ============================================================
// thread_pool.cpp — 轻量级线程池
// 固定数量的工作线程从任务队列中取出异步任务执行。
// 支持 WaitAll 等待所有任务完成，析构时自动清理。
// ============================================================

#include "engine/thread_pool.h"

namespace aicore {

/**
 * 创建线程池，启动指定数量的工作线程
 * 每个线程循环等待条件变量，当 stop_ 标记为 true 且队列为空时退出
 * @param numThreads 工作线程数量
 */
ThreadPool::ThreadPool(size_t numThreads) {
    // ---- 线程池工作模式 ----
    // 采用"领导者-跟随者"（Half-Sync/Half-Async）模式：
    //   领 导 者：工作线程在 std::condition_variable 上等待任务
    //   跟 随 者：提交任务的线程（Enqueue）将任务放入队列并通知领导者
    //
    // 线程生命周期：
    //   每个工作线程在 while(true) 循环中：
    //     1. 获取互斥锁
    //     2. 等待条件变量：队列非空或 stop_ 标记
    //     3. 取出队首任务（在锁保护下）
    //     4. 释放锁
    //     5. 执行任务（在锁外，不阻塞其他线程取任务）
    //     6. 回到步骤 1
    //
    // 为什么在锁外执行任务：
    //   如果在锁内执行任务，会导致：
    //   - 其他线程无法取任务（串行化，退化回单线程）
    //   - 任务中的耗时操作（如模型推理）会阻塞任务分发
    //   这种"取出即释放"的设计是线程池的标准实践。
    //
    // 同步机制：
    //   mutex_ + cv_ (condition_variable) 实现生产者-消费者模型。
    //   工作线程在 cv_.wait() 上阻塞，Enqueue 时 notify_one 唤醒一个线程。
    //   这种唤醒策略比 notify_all 更高效（避免"惊群效应"）。
    //
    // 创建 numThreads 个工作线程，每个线程循环等待任务
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    // 等待条件变量：stop_ 为 true 或队列非空时唤醒
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    // 停止且无剩余任务时退出线程
                    if (stop_ && tasks_.empty()) return;
                    // 取出队首任务
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                // 在锁外执行任务，避免阻塞其他线程
                task();
            }
        });
    }
}

/**
 * 销毁线程池：通知所有线程停止并等待其退出
 */
ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    // 唤醒所有等待的线程
    cv_.notify_all();
    // 等待所有工作线程执行完毕
    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

/**
 * 等待所有已提交的任务执行完毕
 * 当任务队列为空且活跃计数为 0 时返回
 * 注意：WaitAll 期间如果有其他线程 Enqueue，也会等待这些新任务
 */
void ThreadPool::WaitAll() {
    // ---- WaitAll 等待机制 ----
    // 使用 activeCount_ 追踪正在执行的任务数。
    // activeCount_ == 0 时所有任务已完成（tasks_ 受 mutex_ 保护，此处不读取）。
    std::unique_lock<std::mutex> lock(activeMutex_);
    activeCv_.wait(lock, [this] {
        return activeCount_ == 0;
    });
}

/**
 * 获取当前待处理的任务数量
 * @return 队列中等待执行的任务数
 */
size_t ThreadPool::GetPendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

} // namespace aicore
