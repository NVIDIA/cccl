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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__cufile/detail/enums.hpp>

#include <stdexcept>
#include <string>

#include <cufile.h>

// CUDA Experimental cuFILE Library namespace
namespace cuda::experimental::cufile
{
// Forward declare driver API used by helpers below
void driver_open();
void driver_close();
long driver_use_count();
int get_version();

// ================================================================================================
// Error Handling
// ================================================================================================

//! Unified cuFile exception class
class cufile_exception : public ::std::runtime_error
{
private:
  CUfileError_t error_;

public:
  explicit cufile_exception(CUfileError_t error)
      : ::std::runtime_error(format_error_message(error))
      , error_(error)
  {}

  explicit cufile_exception(const ::std::string& message)
      : ::std::runtime_error(message)
      , error_{to_c_enum(cu_file_error::success), CUDA_SUCCESS}
  {}

  CUfileError_t error() const noexcept
  {
    return error_;
  }

private:
  static ::std::string format_error_message(CUfileError_t error)
  {
    return ::std::string("cuFile error: ") + ::std::to_string(error.err) + " (CUDA: " + ::std::to_string(error.cu_err)
         + ")";
  }
};

//! Check cuFile operation result and throw on error
inline void check_cufile_result(CUfileError_t error, const ::std::string& operation = "")
{
  if (error.err != to_c_enum(cu_file_error::success))
  {
    ::std::string message = operation.empty() ? "" : operation + ": ";
    throw cufile_exception(error);
  }
}

//! Check cuFile operation result and throw on error (for ssize_t returns)
inline ssize_t check_cufile_result(ssize_t result, const ::std::string& operation = "")
{
  if (result < 0)
  {
    CUfileError_t error   = {to_c_enum_from_result(result), CUDA_SUCCESS};
    ::std::string message = operation.empty() ? "" : operation + ": ";
    throw cufile_exception(error);
  }
  return result;
}

//! Initialize the cuFILE library
inline void initialize()
{
  driver_open();
}

//! Shutdown the cuFILE library
inline void shutdown()
{
  driver_close();
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

namespace detail
{
inline void check_cufile_result(CUfileError_t error, const ::std::string& operation = "")
{
  ::cuda::experimental::cufile::check_cufile_result(error, operation);
}

inline ssize_t check_cufile_result(ssize_t result, const ::std::string& operation = "")
{
  return ::cuda::experimental::cufile::check_cufile_result(result, operation);
}
} // namespace detail

} // namespace cuda::experimental::cufile

// ================================================================================================
// Core Components
// ================================================================================================

#include <cuda/experimental/__cufile/batch_handle.hpp>
#include <cuda/experimental/__cufile/buffer_handle.hpp>
#include <cuda/experimental/__cufile/driver.hpp>
#include <cuda/experimental/__cufile/file_handle.hpp>
#include <cuda/experimental/__cufile/stream_handle.hpp>
#include <cuda/experimental/__cufile/utils.hpp>
