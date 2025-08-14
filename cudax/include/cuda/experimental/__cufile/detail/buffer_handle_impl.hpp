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

// This file provides the implementation of buffer_handle methods
// It's included after the class definition

#include <cuda/experimental/__cufile/buffer_handle.hpp>

namespace cuda::experimental::cufile
{

// Constructor implementations
template <typename T>
buffer_handle::buffer_handle(cuda::std::span<T> buffer, int flags)
    : buffer_(cuda::std::as_bytes(buffer))
{
  static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  CUfileError_t error = cuFileBufRegister(buffer_.data(), buffer_.size(), flags);
  detail::check_cufile_result(error, "cuFileBufRegister");

  registered_buffer_.emplace(buffer_.data(), [](const void* buf) {
    cuFileBufDeregister(buf);
  });
}

template <typename T>
buffer_handle::buffer_handle(cuda::std::span<const T> buffer, int flags)
    : buffer_(cuda::std::as_bytes(buffer))
{
  static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  CUfileError_t error = cuFileBufRegister(buffer_.data(), buffer_.size(), flags);
  detail::check_cufile_result(error, "cuFileBufRegister");

  registered_buffer_.emplace(buffer_.data(), [](const void* buf) {
    cuFileBufDeregister(buf);
  });
}

// Move constructor and assignment
inline buffer_handle::buffer_handle(buffer_handle&& other) noexcept
    : buffer_(other.buffer_)
    , registered_buffer_(::std::move(other.registered_buffer_))
{}

inline buffer_handle& buffer_handle::operator=(buffer_handle&& other) noexcept
{
  if (this != &other)
  {
    buffer_            = other.buffer_;
    registered_buffer_ = ::std::move(other.registered_buffer_);
  }
  return *this;
}

// Simple getter implementations
inline const void* buffer_handle::data() const noexcept
{
  return buffer_.data();
}

inline size_t buffer_handle::size() const noexcept
{
  return buffer_.size();
}

inline cuda::std::span<const cuda::std::byte> buffer_handle::as_bytes() const noexcept
{
  return buffer_;
}

inline cuda::std::span<cuda::std::byte> buffer_handle::as_writable_bytes() const noexcept
{
  return cuda::std::span<cuda::std::byte>(const_cast<cuda::std::byte*>(buffer_.data()), buffer_.size());
}

// Template method implementations
template <typename T>
cuda::std::span<T> buffer_handle::as_span() const noexcept
{
  static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
  return cuda::std::span<T>(reinterpret_cast<T*>(const_cast<cuda::std::byte*>(buffer_.data())),
                            buffer_.size() / sizeof(T));
}

template <typename T>
cuda::std::span<const T> buffer_handle::as_const_span() const noexcept
{
  static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
  return cuda::std::span<const T>(reinterpret_cast<const T*>(buffer_.data()), buffer_.size() / sizeof(T));
}

// is_valid method implementation
inline bool buffer_handle::is_valid() const noexcept
{
  return registered_buffer_.has_value();
}

} // namespace cuda::experimental::cufile
