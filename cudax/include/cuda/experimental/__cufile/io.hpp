#pragma once

#include "cufile.hpp"
#include "file_handle.hpp"
#include "driver.hpp"
#include "utils.hpp"

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

namespace cuda::experimental {

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

} // namespace cuda::experimental

// Convenience aliases
namespace cufile = cuda::experimental;

using cufile_handle = cuda::experimental::file_handle;
using cufile_buffer = cuda::experimental::buffer_handle;
using cufile_batch = cuda::experimental::batch_handle;
using cufile_stream = cuda::experimental::stream_handle;