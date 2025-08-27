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

#include <cuda/std/chrono>
#include <cuda/std/span>

#include <cuda/experimental/__cufile/cufile.hpp>
#include <cuda/experimental/__cufile/detail/enums.hpp>

#include <functional>
#include <vector>

#include <sys/types.h>

namespace cuda::experimental::cufile
{

//! Batch I/O operation descriptor using span
template <typename T>
struct batch_io_params_span
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  ::cuda::std::span<T> buffer;
  off_t file_offset;
  off_t buffer_offset; // in bytes
  cu_file_opcode opcode; // read or write
  void* cookie; // user data for tracking

  batch_io_params_span(::cuda::std::span<T> buf, off_t f_off, off_t b_off, cu_file_opcode op, void* ck = nullptr)
      : buffer(buf)
      , file_offset(f_off)
      , buffer_offset(b_off)
      , opcode(op)
      , cookie(ck)
  {}
};

//! Batch I/O operation result
struct batch_io_result
{
  void* cookie;
  cu_file_status status;
  size_t result; // bytes transferred or error code

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

//! RAII wrapper for batch operations
class batch_handle
{
private:
  CUfileBatchHandle_t handle_ = nullptr;
  unsigned int max_operations_;

public:
  explicit batch_handle(unsigned int max_operations);

  batch_handle(batch_handle&& other) noexcept;
  batch_handle& operator=(batch_handle&& other) noexcept;

  ~batch_handle() noexcept
  {
    if (handle_ != nullptr)
    {
      cuFileBatchIODestroy(handle_);
      handle_ = nullptr;
    }
  }

  //! Submit batch operations using span
  template <typename T, typename FileHandle>
  void submit(const FileHandle& file_handle_ref,
              ::cuda::std::span<const batch_io_params_span<T>> operations,
              cu_file_batch_submit_flags flags = cu_file_batch_submit_flags::none);

  //! Get batch status
  ::std::vector<batch_io_result> get_status(
    unsigned int min_completed, ::cuda::std::chrono::milliseconds timeout = ::cuda::std::chrono::milliseconds{0});

  //! Cancel batch operations
  void cancel();

  //! Get maximum operations capacity
  unsigned int max_operations() const noexcept;

  //! Check if the handle owns a valid resource
  bool is_valid() const noexcept;
};

// ===================== Inline implementations =====================

inline batch_handle::batch_handle(unsigned int max_operations)
    : max_operations_(max_operations)
{
  CUfileError_t error = cuFileBatchIOSetUp(&handle_, max_operations);
  detail::check_cufile_result(error, "cuFileBatchIOSetUp");
}

inline batch_handle::batch_handle(batch_handle&& other) noexcept
    : handle_(other.handle_)
    , max_operations_(other.max_operations_)
{
  other.handle_ = nullptr;
}

inline batch_handle& batch_handle::operator=(batch_handle&& other) noexcept
{
  if (this != &other)
  {
    if (handle_ != nullptr)
    {
      cuFileBatchIODestroy(handle_);
    }
    handle_         = other.handle_;
    max_operations_ = other.max_operations_;
    other.handle_   = nullptr;
  }
  return *this;
}

inline ::std::vector<batch_io_result>
batch_handle::get_status(unsigned int min_completed, ::cuda::std::chrono::milliseconds timeout)
{
  ::std::vector<CUfileIOEvents_t> events(max_operations_);
  unsigned int num_events = max_operations_;

  timespec timeout_spec;

  if (timeout.count() > 0)
  {
    const auto total_ms  = timeout.count();
    timeout_spec.tv_sec  = total_ms / 1000;
    timeout_spec.tv_nsec = (total_ms % 1000) * 1000000;
  }

  CUfileError_t error = cuFileBatchIOGetStatus(
    handle_, min_completed, &num_events, events.data(), timeout.count() > 0 ? &timeout_spec : nullptr);
  detail::check_cufile_result(error, "cuFileBatchIOGetStatus");

  ::std::vector<batch_io_result> results;
  results.reserve(num_events);

  for (unsigned int i = 0; i < num_events; ++i)
  {
    batch_io_result result = {};
    result.cookie          = events[i].cookie;
    result.status          = to_cpp_enum(events[i].status);
    result.result          = events[i].ret;
    results.push_back(result);
  }

  return results;
}

inline void batch_handle::cancel()
{
  CUfileError_t error = cuFileBatchIOCancel(handle_);
  detail::check_cufile_result(error, "cuFileBatchIOCancel");
  handle_ = nullptr;
}

inline unsigned int batch_handle::max_operations() const noexcept
{
  return max_operations_;
}

inline bool batch_handle::is_valid() const noexcept
{
  return handle_ != nullptr;
}

// Generic submit<T, Handle> implementation (Handle must provide native_handle())
template <typename T, typename Handle>
inline void batch_handle::submit(const Handle& file_handle_ref,
                                 ::cuda::std::span<const batch_io_params_span<T>> operations,
                                 cu_file_batch_submit_flags flags)
{
  ::std::vector<CUfileIOParams_t> cufile_ops;
  cufile_ops.reserve(operations.size());

  for (const auto& op : operations)
  {
    auto& cufile_op                 = cufile_ops.emplace_back();
    cufile_op.mode                  = to_c_enum(cu_file_mode::batch);
    cufile_op.u.batch.devPtr_base   = op.buffer.data();
    cufile_op.u.batch.file_offset   = op.file_offset;
    cufile_op.u.batch.devPtr_offset = op.buffer_offset;
    cufile_op.u.batch.size          = op.buffer.size_bytes();
    cufile_op.fh                    = file_handle_ref.native_handle();
    cufile_op.opcode                = to_c_enum(op.opcode);
    cufile_op.cookie                = op.cookie;
  }

  CUfileError_t error = cuFileBatchIOSubmit(handle_, cufile_ops.size(), cufile_ops.data(), to_c_enum(flags));
  detail::check_cufile_result(error, "cuFileBatchIOSubmit");
}

// Free function template implementations
template <typename T>
batch_io_params_span<T>
make_read_operation(::cuda::std::span<T> buffer, off_t file_offset, off_t buffer_offset, void* cookie)
{
  return batch_io_params_span<T>(buffer, file_offset, buffer_offset, cu_file_opcode::read, cookie);
}

template <typename T>
batch_io_params_span<const T>
make_write_operation(::cuda::std::span<const T> buffer, off_t file_offset, off_t buffer_offset, void* cookie)
{
  return batch_io_params_span<const T>(buffer, file_offset, buffer_offset, cu_file_opcode::write, cookie);
}

// Free function templates are defined below; no separate declarations needed

} // namespace cuda::experimental::cufile
