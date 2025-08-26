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

#include <cuda/experimental/__cufile/detail/error_handling.hpp>
#include <cuda/experimental/__cufile/detail/raii_resource.hpp>

#include <functional>

namespace cuda::experimental::cufile
{

/**
 * @brief RAII wrapper for GPU buffer registration
 */
class buffer_handle
{
private:
  ::cuda::std::span<const ::cuda::std::byte> buffer_;
  detail::raii_resource<const void*, void (*)(const void*)> registered_buffer_;

public:
  /**
   * @brief Register GPU buffer using span
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the GPU buffer
   * @param flags Registration flags (default: 0)
   */
  template <typename T>
  explicit buffer_handle(::cuda::std::span<T> buffer, int flags = 0);

  /**
   * @brief Register GPU buffer using span - const version
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the GPU buffer
   * @param flags Registration flags (default: 0)
   */
  template <typename T>
  explicit buffer_handle(::cuda::std::span<const T> buffer, int flags = 0);

  buffer_handle(buffer_handle&& other) noexcept;
  buffer_handle& operator=(buffer_handle&& other) noexcept;

  /**
   * @brief Get the registered buffer pointer
   */
  const void* data() const noexcept;

  /**
   * @brief Get the buffer size in bytes
   */
  size_t size() const noexcept;

  /**
   * @brief Get the buffer as a span of bytes
   */
  ::cuda::std::span<const ::cuda::std::byte> as_bytes() const noexcept;

  /**
   * @brief Get the buffer as a span of mutable bytes
   */
  ::cuda::std::span<::cuda::std::byte> as_writable_bytes() const noexcept;

  /**
   * @brief Get the buffer as a typed span
   * @tparam T Element type (must be trivially copyable)
   * @return Span of type T over the buffer
   */
  template <typename T>
  ::cuda::std::span<T> as_span() const noexcept;

  /**
   * @brief Get the buffer as a typed const span
   * @tparam T Element type (must be trivially copyable)
   * @return Const span of type T over the buffer
   */
  template <typename T>
  ::cuda::std::span<const T> as_const_span() const noexcept;

  /**
   * @brief Check if the handle owns a valid resource
   */
  bool is_valid() const noexcept;
};

// ===================== Inline implementations =====================

// Constructor implementations
template <typename T>
inline buffer_handle::buffer_handle(::cuda::std::span<T> buffer, int flags)
    : buffer_(::cuda::std::as_bytes(buffer))
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

  CUfileError_t error = cuFileBufRegister(buffer_.data(), buffer_.size(), flags);
  detail::check_cufile_result(error, "cuFileBufRegister");

  registered_buffer_.emplace(buffer_.data(), [](const void* buf) {
    cuFileBufDeregister(buf);
  });
}

template <typename T>
inline buffer_handle::buffer_handle(::cuda::std::span<const T> buffer, int flags)
    : buffer_(::cuda::std::as_bytes(buffer))
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

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

inline ::cuda::std::span<const ::cuda::std::byte> buffer_handle::as_bytes() const noexcept
{
  return buffer_;
}

inline ::cuda::std::span<::cuda::std::byte> buffer_handle::as_writable_bytes() const noexcept
{
  return ::cuda::std::span<::cuda::std::byte>(const_cast<::cuda::std::byte*>(buffer_.data()), buffer_.size());
}

// Template method implementations
template <typename T>
inline ::cuda::std::span<T> buffer_handle::as_span() const noexcept
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
  return ::cuda::std::span<T>(
    reinterpret_cast<T*>(const_cast<::cuda::std::byte*>(buffer_.data())), buffer_.size() / sizeof(T));
}

template <typename T>
inline ::cuda::std::span<const T> buffer_handle::as_const_span() const noexcept
{
  static_assert(::cuda::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
  return ::cuda::std::span<const T>(reinterpret_cast<const T*>(buffer_.data()), buffer_.size() / sizeof(T));
}

inline bool buffer_handle::is_valid() const noexcept
{
  return registered_buffer_.has_value();
}

} // namespace cuda::experimental::cufile
