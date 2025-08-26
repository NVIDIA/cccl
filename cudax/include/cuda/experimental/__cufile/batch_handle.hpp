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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/span>

#include <cuda/experimental/__cufile/detail/enums.hpp>
#include <cuda/experimental/__cufile/detail/error_handling.hpp>
#include <cuda/experimental/__cufile/detail/raii_resource.hpp>

#include <functional>
#include <vector>

#include <sys/types.h>

namespace cuda::experimental::cufile
{

// Forward declarations
class file_handle_base;

//! @brief Batch I/O operation descriptor using span
//! @tparam T Element type (must be trivially copyable)
template <typename T>
struct batch_io_params_span
{
  ::cuda::std::span<T> buffer; ///< Buffer span
  off_t file_offset; ///< File offset
  off_t buffer_offset; ///< Buffer offset (in bytes)
  cu_file_opcode opcode; ///< cuFile operation code (read or write)
  void* cookie; ///< User data for tracking

  // Constructor
  batch_io_params_span(::cuda::std::span<T> buf, off_t f_off, off_t b_off, cu_file_opcode op, void* ck = nullptr)
      : buffer(buf)
      , file_offset(f_off)
      , buffer_offset(b_off)
      , opcode(op)
      , cookie(ck)
  {
    static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
  }
};

/**
 * @brief Batch I/O operation result
 */
struct batch_io_result
{
  void* cookie; ///< User data from operation
  cu_file_status status; ///< Operation status
  size_t result; ///< Bytes transferred or error code

  bool is_complete() const noexcept
  {
    return status == cu_file_status::complete;
  }
  bool is_failed() const noexcept
  {
    return status == cu_file_status::failed;
  }
  bool has_error() const noexcept
  {
    return static_cast<ssize_t>(result) < 0;
  }
};

/**
 * @brief RAII wrapper for batch operations
 */
class batch_handle
{
private:
  CUfileBatchHandle_t handle_;
  unsigned int max_operations_;
  detail::raii_resource<CUfileBatchHandle_t, ::std::function<void(CUfileBatchHandle_t)>> batch_resource_;

public:
  /**
   * @brief Create batch handle
   * @param max_operations Maximum number of operations
   */
  explicit batch_handle(unsigned int max_operations);

  batch_handle(batch_handle&& other) noexcept;
  batch_handle& operator=(batch_handle&& other) noexcept;

  /**
   * @brief Submit batch operations using span
   * @tparam T Element type (must be trivially copyable)
   * @param file_handle_ref File handle to operate on
   * @param operations Span of span-based batch operations
   * @param flags Additional flags (default: none)
   */
  template <typename T>
  void submit(const file_handle_base& file_handle_ref,
              ::cuda::std::span<const batch_io_params_span<T>> operations,
              cu_file_batch_submit_flags flags = cu_file_batch_submit_flags::none);

  /**
   * @brief Get batch status
   */
  ::std::vector<batch_io_result> get_status(unsigned int min_completed, int timeout_ms = 0);

  /**
   * @brief Cancel batch operations
   */
  void cancel();

  /**
   * @brief Get maximum operations capacity
   */
  unsigned int max_operations() const noexcept;

  /**
   * @brief Check if the handle owns a valid resource
   */
  bool is_valid() const noexcept;
};

/**
 * @brief Create a read operation for batch processing
 * @tparam T Element type
 * @param buffer Buffer span to read into
 * @param file_offset File offset to read from
 * @param buffer_offset Buffer offset (in bytes)
 * @param cookie User data for tracking
 */
template <typename T>
batch_io_params_span<T>
make_read_operation(::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset = 0, void* cookie = nullptr);

/**
 * @brief Create a write operation for batch processing
 * @tparam T Element type
 * @param buffer Buffer span to write from
 * @param file_offset File offset to write to
 * @param buffer_offset Buffer offset (in bytes)
 * @param cookie User data for tracking
 */
template <typename T>
batch_io_params_span<const T> make_write_operation(
  ::cuda::std::span<const T> buffer, off_t file_offset, off_t buffer_offset = 0, void* cookie = nullptr);

} // namespace cuda::experimental::cufile
