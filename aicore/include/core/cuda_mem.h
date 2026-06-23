#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace aicore {

class CudaMem {
public:
    CudaMem() = default;

    cudaError_t Alloc(size_t bytes) {
        Free();
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err == cudaSuccess) bytes_ = bytes;
        return err;
    }

    void Free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    ~CudaMem() { Free(); }

    CudaMem(CudaMem&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

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

    CudaMem(const CudaMem&) = delete;
    CudaMem& operator=(const CudaMem&) = delete;

    float* Get() const { return ptr_; }
    size_t Bytes() const { return bytes_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    float* ptr_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace aicore
