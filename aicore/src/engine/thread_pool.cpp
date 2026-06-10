#include "engine/thread_pool.h"

namespace aicore {

ThreadPool::ThreadPool(size_t numThreads) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

void ThreadPool::WaitAll() {
    std::unique_lock<std::mutex> lock(activeMutex_);
    activeCv_.wait(lock, [this] {
        return tasks_.empty() && activeCount_ == 0;
    });
}

size_t ThreadPool::GetPendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

} // namespace aicore
