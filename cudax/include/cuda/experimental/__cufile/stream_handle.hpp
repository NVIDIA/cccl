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

#include <cuda/experimental/__cufile/cufile.hpp>

#include <functional>

namespace cuda::experimental::cufile
{

//! RAII wrapper for CUDA stream registration with cuFILE
class stream_handle
{
private:
  cudaStream_t registered_stream_ = nullptr;

public:
  //! Register CUDA stream
  stream_handle(cuda::stream_ref stream, unsigned int flags = 0);

  stream_handle(stream_handle&& other) noexcept;
  stream_handle& operator=(stream_handle&& other) noexcept;

  ~stream_handle() noexcept;

  //! Get the registered CUDA stream
  cudaStream_t get() const noexcept;

  //! Check if the handle owns a valid resource
  bool is_valid() const noexcept;
};

// ===================== Inline implementations =====================

inline stream_handle::stream_handle(cuda::stream_ref stream, unsigned int flags)
{
  CUfileError_t error = cuFileStreamRegister(stream.get(), flags);
  detail::check_cufile_result(error, "cuFileStreamRegister");
  registered_stream_ = stream.get();
}

inline stream_handle::stream_handle(stream_handle&& other) noexcept
    : registered_stream_(other.registered_stream_)
{
  other.registered_stream_ = nullptr;
}

inline stream_handle& stream_handle::operator=(stream_handle&& other) noexcept
{
  if (this != &other)
  {
    if (registered_stream_ != nullptr)
    {
      cuFileStreamDeregister(registered_stream_);
    }
    registered_stream_       = other.registered_stream_;
    other.registered_stream_ = nullptr;
  }
  return *this;
}

inline stream_handle::~stream_handle() noexcept
{
  if (registered_stream_ != nullptr)
  {
    cuFileStreamDeregister(registered_stream_);
    registered_stream_ = nullptr;
  }
}

inline cudaStream_t stream_handle::get() const noexcept
{
  return registered_stream_;
}

inline bool stream_handle::is_valid() const noexcept
{
  return registered_stream_ != nullptr;
}

} // namespace cuda::experimental::cufile
