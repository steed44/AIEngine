#pragma once
#include "core/types.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>

namespace aicore {

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();

    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>;

    void WaitAll();
    size_t GetThreadCount() const { return workers_.size(); }
    size_t GetPendingCount() const;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    size_t activeCount_ = 0;
    mutable std::mutex activeMutex_;
    std::condition_variable activeCv_;
};

template<class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result_t<F, Args...>>
{
    using ReturnType = typename std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<ReturnType> result = task->get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_)
            throw std::runtime_error("ThreadPool: enqueue on stopped pool");
        tasks_.emplace([task, this]() {
            {
                std::lock_guard<std::mutex> al(activeMutex_);
                ++activeCount_;
            }
            (*task)();
            {
                std::lock_guard<std::mutex> al(activeMutex_);
                --activeCount_;
            }
            activeCv_.notify_all();
        });
    }
    cv_.notify_one();
    return result;
}

} // namespace aicore
