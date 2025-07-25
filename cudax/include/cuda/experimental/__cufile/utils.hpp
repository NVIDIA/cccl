#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda::experimental {

namespace utils {

/**
 * @brief Check if a pointer is GPU memory
 */
inline bool is_gpu_memory(const void* ptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    return (err == cudaSuccess) && (attrs.type == cudaMemoryTypeDevice);
}

/**
 * @brief Get device ID for a GPU memory pointer
 */
inline int get_device_id(const void* ptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get pointer attributes");
    }
    return attrs.device;
}

/**
 * @brief Check if size is properly aligned
 */
inline bool is_aligned(size_t size, size_t alignment) {
    return (size % alignment) == 0;
}

/**
 * @brief Round up to nearest alignment boundary
 */
inline size_t align_up(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * @brief Get optimal alignment for cuFile operations
 */
inline size_t get_optimal_alignment() {
    return 4096; // 4KB alignment for most file systems
}

/**
 * @brief Check if a pointer is suitable for cuFile operations
 */
inline bool is_cufile_compatible(const void* ptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        return false;
    }

    // cuFile works with device memory and registered host memory
    return (attrs.type == cudaMemoryTypeDevice) ||
           (attrs.type == cudaMemoryTypeHost);
}

} // namespace utils

} // namespace cuda::experimental