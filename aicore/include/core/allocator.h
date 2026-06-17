// 内存分配器抽象接口
// 统一管理 CPU/GPU 内存的分配、释放和拷贝操作
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
    // @param bytes 请求的字节数
    // @param type  内存类型（CPU/GPU 等）
    // @return 分配的内存指针，失败返回 nullptr
    virtual void* Alloc(size_t bytes, MemoryType type) = 0;
    // 释放指定类型的内存
    // @param ptr  待释放的内存指针
    // @param type 内存类型（必须与分配时一致）
    virtual void Free(void* ptr, MemoryType type) = 0;
    // 在设备之间拷贝内存数据
    // @param dst     目标内存指针
    // @param src     源内存指针
    // @param bytes   拷贝字节数
    // @param dstType 目标内存类型
    // @param srcType 源内存类型
    // @param stream  CUDA 流（可选，用于异步拷贝）
    // @return 成功返回 Status::kOk
    virtual Status Copy(void* dst, const void* src,
                        size_t bytes, MemoryType dstType,
                        MemoryType srcType, CudaStream stream = nullptr) = 0;
    // 获取当前已分配的总字节数（用于内存泄漏检测和统计）
    // @return 已分配字节数
    virtual size_t GetAllocatedBytes() const = 0;
};

} // namespace aicore
