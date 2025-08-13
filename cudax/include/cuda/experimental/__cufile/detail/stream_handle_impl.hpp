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

// This file provides the implementation of stream_handle methods
// It's included after the class definition
#include "cuda/experimental/__cufile/stream_handle.hpp"

namespace cuda::experimental::cufile
{

// Constructor implementation
inline stream_handle::stream_handle(cudaStream_t stream, unsigned int flags)
    : stream_(stream)
{
  CUfileError_t error = cuFileStreamRegister(stream, flags);
  detail::check_cufile_result(error, "cuFileStreamRegister");

  registered_stream_.emplace(stream, [](cudaStream_t s) {
    cuFileStreamDeregister(s);
  });
}

// Move constructor and assignment
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

// Simple getter implementation
inline cudaStream_t stream_handle::get() const noexcept
{
  return stream_;
}

inline bool stream_handle::is_valid() const noexcept
{
  return registered_stream_.has_value();
}

} // namespace cuda::experimental::cufile
