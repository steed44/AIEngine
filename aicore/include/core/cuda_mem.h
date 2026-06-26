// CUDA GPU 显存 RAII 封装
// 自动管理 cudaMalloc/cudaFree 生命周期，防止显存泄漏
//
// 设计模式：RAII（资源获取即初始化）
// 线程安全：非线程安全，每个 CudaMem 实例应由单一线程使用
// 所有权：独占所有权，不支持拷贝语义，仅支持移动语义
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace aicore {

// GPU 显存管理器（RAII）
// 构造时分配，析构时自动释放；支持 move 语义但不支持 copy
//
// 前置条件：调用 Alloc 前应确保当前 CUDA 上下文已初始化
// 后置条件：析构时自动释放持有的显存资源
// 设计说明：选择 float* 而非 void* 是因为大部分场景操作 float 张量，
//           减少外部类型转换。需要其他类型时 reinterpret_cast 使用
class CudaMem {
public:
    CudaMem() = default;

    // 分配指定字节数的 GPU 显存
    // @param bytes 要分配的字节数
    // @return cudaError_t，cudaSuccess 表示成功
    // 前置条件：当前 CUDA 上下文有效，bytes > 0
    // 后置条件：成功时 ptr_ 指向新分配的显存，失败时 ptr_ 保持不变
    cudaError_t Alloc(size_t bytes) {
        Free();
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err == cudaSuccess) bytes_ = bytes;
        return err;
    }

    // 释放显存并重置指针
    // 后置条件：ptr_ == nullptr, bytes_ == 0
    // 幂等性：可多次调用，第二次调用无效果
    void Free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    // 析构时自动释放显存
    ~CudaMem() { Free(); }

    // 移动构造函数：转移显存所有权，源对象置空
    // 后置条件：other.ptr_ == nullptr, other.bytes_ == 0
    CudaMem(CudaMem&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    // 移动赋值运算符：释放当前显存后转移所有权
    // 前置条件：无（自赋值安全）
    // 后置条件：原资源释放，新资源接管
    CudaMem& operator=(CudaMem&& other) noexcept {
        if (this != &other) {
            Free();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    // 禁止拷贝语义：显存资源不可隐式复制
    CudaMem(const CudaMem&) = delete;
    CudaMem& operator=(const CudaMem&) = delete;

    float* Get() const { return ptr_; }       // 获取显存指针
    size_t Bytes() const { return bytes_; }   // 获取分配大小（字节）
    explicit operator bool() const { return ptr_ != nullptr; }  // 检查是否已分配

private:
    float* ptr_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace aicore
