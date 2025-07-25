#pragma once

/**
 * @file cufile.hpp
 * @brief Modern C++ bindings for NVIDIA cuFILE (GPU Direct Storage)
 *
 * Provides clean, modern C++ interface that directly maps to the cuFILE C API.
 */

// ================================================================================================
// Core Components
// ================================================================================================

#include "file_handle.hpp"     // File operations
#include "driver.hpp"          // Driver management and configuration
#include "utils.hpp"           // Utility functions

/**
 * @namespace cuda::experimental::cufile
 * @brief CUDA Experimental cuFILE Library namespace
 */
namespace cuda::experimental::cufile {

// ================================================================================================
// Error Handling
// ================================================================================================

using cufile_exception = detail::cufile_exception;

/**
 * @brief Initialize the cuFILE library
 *
 * Must be called before using any cuFILE operations. Use driver_handle for RAII management.
 */
inline void initialize() {
    driver_open();
}

/**
 * @brief Shutdown the cuFILE library
 *
 * Should be called after all cuFILE operations are complete.
 */
inline void shutdown() noexcept {
    try { driver_close(); } catch(...) {}
}

/**
 * @brief Check if the cuFILE library is initialized
 */
inline bool is_initialized() noexcept {
    return driver_use_count() > 0;
}

/**
 * @brief Get cuFILE library version information
 */
inline int get_cufile_version() noexcept {
    return get_version();
}

} // namespace cuda::experimental::cufile