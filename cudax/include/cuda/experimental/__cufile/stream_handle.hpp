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

#include <cuda/experimental/__cufile/detail/error_handling.hpp>
#include <cuda/experimental/__cufile/detail/raii_resource.hpp>

#include <functional>

namespace cuda::experimental::cufile
{

/**
 * @brief RAII wrapper for CUDA stream registration with cuFILE
 */
class stream_handle
{
private:
  cudaStream_t stream_;
  detail::raii_resource<cudaStream_t, ::std::function<void(cudaStream_t)>> registered_stream_;

public:
  /**
   * @brief Register CUDA stream
   * @param stream CUDA stream to register
   * @param flags Stream flags (see CU_FILE_STREAM_* constants)
   */
  stream_handle(cudaStream_t stream, unsigned int flags = 0);

  stream_handle(stream_handle&& other) noexcept;
  stream_handle& operator=(stream_handle&& other) noexcept;

  /**
   * @brief Get the registered CUDA stream
   */
  cudaStream_t get() const noexcept;

  /**
   * @brief Check if the handle owns a valid resource
   */
  bool is_valid() const noexcept;
};

// ===================== Inline implementations =====================

inline stream_handle::stream_handle(cudaStream_t stream, unsigned int flags)
    : stream_(stream)
{
  CUfileError_t error = cuFileStreamRegister(stream, flags);
  detail::check_cufile_result(error, "cuFileStreamRegister");

  registered_stream_.emplace(stream, [](cudaStream_t s) {
    cuFileStreamDeregister(s);
  });
}

inline stream_handle::stream_handle(stream_handle&& other) noexcept
    : stream_(other.stream_)
    , registered_stream_(::std::move(other.registered_stream_))
{}

inline stream_handle& stream_handle::operator=(stream_handle&& other) noexcept
{
  if (this != &other)
  {
    stream_            = other.stream_;
    registered_stream_ = ::std::move(other.registered_stream_);
  }
  return *this;
}

inline cudaStream_t stream_handle::get() const noexcept
{
  return stream_;
}

inline bool stream_handle::is_valid() const noexcept
{
  return registered_stream_.has_value();
}

} // namespace cuda::experimental::cufile
