#include "test_utils.h"
#include <iostream>
#include <cstdlib>  // For malloc and free

// Try to include CUDA headers, but handle the case where they're not available
#ifdef __has_include
    #if __has_include(<cuda_runtime.h>)
        #include <cuda_runtime.h>
        #define CUDA_AVAILABLE 1
    #else
        #define CUDA_AVAILABLE 0
    #endif
#else
    // Fallback for older compilers
    #include <cuda_runtime.h>
    #define CUDA_AVAILABLE 1
#endif

namespace cuda::experimental::cufile::test_utils {

bool is_cuda_available() {
#if CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

void* allocate_gpu_memory(size_t size) {
#if CUDA_AVAILABLE
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(error)));
    }
    return ptr;
#else
    (void)size;  // Suppress unused parameter warning
    throw std::runtime_error("CUDA not available - cannot allocate GPU memory");
#endif
}

void* allocate_host_memory(size_t size) {
#if CUDA_AVAILABLE
    void* ptr = nullptr;
    cudaError_t error = cudaMallocHost(&ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host memory: " + std::string(cudaGetErrorString(error)));
    }
    return ptr;
#else
    (void)size;  // Suppress unused parameter warning
    throw std::runtime_error("CUDA not available - cannot allocate pinned host memory");
#endif
}

void free_gpu_memory(void* ptr) {
#if CUDA_AVAILABLE
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
#else
    (void)ptr;  // Suppress unused parameter warning
#endif
}

void free_host_memory(void* ptr) {
#if CUDA_AVAILABLE
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
#else
    (void)ptr;  // Suppress unused parameter warning
#endif
}

void* allocate_regular_memory(size_t size) {
    return malloc(size);
}

void free_regular_memory(void* ptr) {
    if (ptr != nullptr) {
        free(ptr);
    }
}

GPUMemoryRAII::GPUMemoryRAII(size_t size) : ptr_(nullptr), size_(size) {
    try {
        ptr_ = allocate_gpu_memory(size);
    } catch (const std::exception& e) {
        // Re-throw with more context
        throw std::runtime_error("GPUMemoryRAII constructor failed: " + std::string(e.what()));
    }
}

GPUMemoryRAII::~GPUMemoryRAII() {
    free_gpu_memory(ptr_);
}

void* GPUMemoryRAII::get() const {
    return ptr_;
}

size_t GPUMemoryRAII::size() const {
    return size_;
}

HostMemoryRAII::HostMemoryRAII(size_t size) : ptr_(nullptr), size_(size) {
    try {
        ptr_ = allocate_host_memory(size);
    } catch (const std::exception& e) {
        // Re-throw with more context
        throw std::runtime_error("HostMemoryRAII constructor failed: " + std::string(e.what()));
    }
}

HostMemoryRAII::~HostMemoryRAII() {
    free_host_memory(ptr_);
}

void* HostMemoryRAII::get() const {
    return ptr_;
}

size_t HostMemoryRAII::size() const {
    return size_;
}

RegularMemoryRAII::RegularMemoryRAII(size_t size) : ptr_(nullptr), size_(size) {
    ptr_ = allocate_regular_memory(size);
}

RegularMemoryRAII::~RegularMemoryRAII() {
    free_regular_memory(ptr_);
}

void* RegularMemoryRAII::get() const {
    return ptr_;
}

size_t RegularMemoryRAII::size() const {
    return size_;
}

} // namespace cuda::experimental::cufile::test_utils