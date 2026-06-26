// GPU 显存管理器 — 带 LRU 驱逐策略的内存池
// 单例模式，管理所有 GPU 显存分配/释放/回收
//
// 设计模式：单例（Singleton）+ LRU 缓存
// 线程安全：Alloc/Free/Touch 全部互斥保护
// 所有权：显存由管理器统一管理，外部通过 allocId 追踪生命周期
// 预算策略：默认上限为 GPU 总显存的 70%，超出时驱逐最久未访问的块
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace aicore {

// GPU 显存管理器
// 提供带预算上限的显存分配（默认 GPU 总显存 70%）
// 超出预算时按 LRU（最近最少使用）策略驱逐已有分配
//
// 使用示例：
//   auto& mgr = MemoryManager::GetInstance();
//   uint64_t id;
//   float* ptr = mgr.TryAlloc(1024 * 1024, id);
//   mgr.Touch(id);  // 标记最近使用，降低 LRU 驱逐优先级
//   mgr.Free(id);   // 主动释放
class MemoryManager {
public:
    static MemoryManager& GetInstance();

    // 尝试分配显存，超预算时自动 LRU 驱逐
    // @return 成功返回指针，失败返回 nullptr
    float* TryAlloc(size_t bytes, uint64_t& outAllocId);
    // 释放指定 ID 的显存
    void Free(uint64_t allocId);
    // 更新访问时间戳（LRU 换出优先级降低）
    void Touch(uint64_t allocId);

    void SetBudget(size_t maxBytes);
    size_t Budget() const { return budget_; }
    size_t Used() const { return used_; }
    size_t Available() const;

    // 查询当前 GPU 的总显存大小
    static size_t QueryTotalVRAM();

private:
    MemoryManager();
    ~MemoryManager();
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // LRU 驱逐：按最后访问时间排序，驱逐最旧的分配直到释放足够空间
    bool EvictLRU(size_t bytesNeeded);

    struct AllocRecord {
        uint64_t id;           // 分配 ID
        float* ptr;            // 显存指针
        size_t bytes;          // 分配大小
        uint64_t lastTouch;    // 最后访问时间戳（用于 LRU 排序）
    };

    std::mutex mutex_;                          // 线程安全互斥锁
    size_t budget_;                             // 显存预算上限
    size_t used_ = 0;                           // 当前已使用量
    uint64_t clock_ = 0;                        // 全局逻辑时钟（用于 LRU 时间戳）
    uint64_t nextId_ = 1;                       // 下一个分配 ID
    std::map<uint64_t, AllocRecord> allocs_;    // 分配记录表
};

} // namespace aicore
