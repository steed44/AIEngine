// 通用线程池
// 管理一组工作线程，支持异步任务投递、同步等待和并行执行
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

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 通用线程池实现
// 支持任意可调用对象的异步执行，通过 std::future 获取结果
// 跟踪活跃任务数量，提供 WaitAll 方法等待所有任务完成
class ThreadPool {
public:
    // 构造指定数量的工作线程
    // @param numThreads 线程数，默认使用硬件并发数
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    // 析构函数：设置停止标志，等待所有线程退出
    ~ThreadPool();

    // 向线程池投递异步任务
    // @param f     可调用对象（函数、lambda 等）
    // @param args  调用参数
    // @return std::future 对象，可通过 .get() 获取返回值
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>;

    // 等待所有已投递的任务执行完毕
    void WaitAll();
    // 获取工作线程数量
    // @return 线程数
    size_t GetThreadCount() const noexcept { return workers_.size(); }
    // 获取等待队列中的待处理任务数量
    // @return 待处理任务数
    size_t GetPendingCount() const;

private:
    std::vector<std::thread> workers_;           // 工作线程集合
    std::queue<std::function<void()>> tasks_;    // 待执行任务队列
    mutable std::mutex mutex_;                   // 任务队列互斥锁
    std::condition_variable cv_;                 // 任务到达通知条件变量
    bool stop_ = false;                          // 停止标志
    size_t activeCount_ = 0;                     // 当前活跃任务数
    mutable std::mutex activeMutex_;             // 活跃计数互斥锁
    std::condition_variable activeCv_;           // 活跃任务完成通知变量
};

// 模板方法 Enqueue 实现
// 将可调用对象和参数打包为 packaged_task，放入任务队列等待工作线程执行
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
