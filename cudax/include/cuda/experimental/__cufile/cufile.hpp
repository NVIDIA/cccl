//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

// cufile.hpp — Modern C++ bindings for NVIDIA cuFILE (GPU Direct Storage)
// Provides a clean interface that directly maps to the cuFILE C API.

// ================================================================================================
// Core Components
// ================================================================================================

#include "driver.hpp" // Driver management and configuration
#include "file_handle.hpp" // File operations
#include "utils.hpp" // Utility functions

// CUDA Experimental cuFILE Library namespace
namespace cuda::experimental::cufile
{

// ================================================================================================
// Error Handling
// ================================================================================================

using cufile_exception = detail::cufile_exception;

//! Initialize the cuFILE library
inline void initialize()
{
  driver_open();
}

//! Shutdown the cuFILE library
inline void shutdown() noexcept
{
  try
  {
    driver_close();
  }
  catch (...)
  {}
}

//! Check if the cuFILE library is initialized
inline bool is_initialized() noexcept
{
  return driver_use_count() > 0;
}

//! Get cuFILE library version information
inline int get_cufile_version() noexcept
{
  return get_version();
}

} // namespace cuda::experimental::cufile
