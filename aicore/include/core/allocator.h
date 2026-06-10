#pragma once
#include "core/types.h"
#include <vector>

namespace aicore {

using CudaStream = void*;

class IAllocator {
public:
    virtual ~IAllocator() = default;

    virtual void* Alloc(size_t bytes, MemoryType type) = 0;
    virtual void Free(void* ptr, MemoryType type) = 0;
    virtual Status Copy(void* dst, const void* src,
                        size_t bytes, MemoryType dstType,
                        MemoryType srcType, CudaStream stream = nullptr) = 0;
    virtual size_t GetAllocatedBytes() const = 0;
};

} // namespace aicore
