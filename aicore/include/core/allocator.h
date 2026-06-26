// 内存分配器抽象接口
// 统一管理 CPU/GPU 内存的分配、释放和拷贝操作
//
// 设计模式：抽象工厂（Abstract Factory）
//   IAllocator 定义内存操作的统一接口，不同后端（CUDA/Pinned/CPU）
//   各自实现此接口。上层代码通过此接口管理所有内存，无需关心底层实现
//
// 线程安全：实现应保证 Alloc/Free/Copy 的线程安全性
//   （内部通过互斥锁或原子操作实现）
#pragma once
#include "core/types.h"
#include <vector>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// CUDA 流句柄，用于异步内存操作
using CudaStream = void*;

// 内存分配器的抽象接口
// 支持不同内存类型（CPU 页锁定内存、GPU 显存等）的分配和跨设备拷贝
class IAllocator {
public:
    virtual ~IAllocator() = default;

    // 分配指定类型的内存
    // @param bytes 请求的字节数（必须 > 0）
    // @param type  内存类型（kCPU/kGPU/kPinned）
    // @return 分配的内存指针，失败返回 nullptr
    // 前置条件：bytes > 0
    // 后置条件：成功时返回非空指针，内存已清零（实现定义）
    virtual void* Alloc(size_t bytes, MemoryType type) = 0;
    // 释放指定类型的内存
    // @param ptr  待释放的内存指针（必须非空，必须来自 Alloc）
    // @param type 内存类型（必须与分配时一致）
    // 前置条件：ptr 非空且由 Alloc 分配
    // 后置条件：ptr 指向的内存已释放，不可再访问
    virtual void Free(void* ptr, MemoryType type) = 0;
    // 在设备之间拷贝内存数据
    // 支持同设备拷贝（CPU→CPU、GPU→GPU）和跨设备拷贝（CPU↔GPU）
    // @param dst     目标内存指针（必须已分配且大小 >= bytes）
    // @param src     源内存指针
    // @param bytes   拷贝字节数
    // @param dstType 目标内存类型
    // @param srcType 源内存类型
    // @param stream  CUDA 流（可选，用于异步 GPU 拷贝）
    // @return 成功返回 Status::kOk
    // 前置条件：dst 和 src 非空，区域内不重叠
    // 线程安全：同一 stream 上并发 Copy 需外部同步
    virtual Status Copy(void* dst, const void* src,
                        size_t bytes, MemoryType dstType,
                        MemoryType srcType, CudaStream stream = nullptr) = 0;
    // 获取当前已分配的总字节数（用于内存泄漏检测和统计）
    // @return 已分配字节数（Alloc - Free 的差值）
    // 线程安全：const 方法，线程安全
    virtual size_t GetAllocatedBytes() const = 0;
};

} // namespace aicore
