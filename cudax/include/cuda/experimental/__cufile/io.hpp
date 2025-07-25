#pragma once

#include <cuda/io/cufile.hpp>
#include <cuda/io/file_handle.hpp>
#include <cuda/io/driver.hpp>
#include <cuda/io/utils.hpp>

/**
 * @file io.hpp
 * @brief Complete CUDA I/O Library - Modern C++ bindings for NVIDIA cuFILE
 * 
 * Provides complete access to all cuFILE functionality through modern C++ interfaces.
 * 
 * Core features:
 * - Direct file operations with RAII resource management
 * - Batch I/O operations for high throughput
 * - Stream management for async operations
 * - Driver configuration and capability detection
 * - Strong type safety with zero-cost abstractions
 */

namespace cuda::io {

/**
 * @brief Initialize the complete cuFILE library
 * 
 * Must be called before using any cuFILE operations. Use driver_handle for RAII management.
 */
inline void initialize() {
    driver_open();
}

/**
 * @brief Check if the library is fully initialized
 */
inline bool is_initialized() noexcept {
    return driver_use_count() > 0;
}

} // namespace cuda::io

// Convenience aliases
namespace cufile = cuda::io;

using cufile_handle = cuda::io::file_handle;
using cufile_buffer = cuda::io::buffer_handle;
using cufile_batch = cuda::io::batch_handle;
using cufile_stream = cuda::io::stream_handle; 