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

#include <functional>

#include "detail/error_handling.hpp"
#include "detail/raii_resource.hpp"

namespace cuda::experimental::cufile
{

/**
 * @brief RAII wrapper for GPU buffer registration
 */
class buffer_handle
{
private:
  cuda::std::span<const cuda::std::byte> buffer_;
  detail::raii_resource<const void*, ::std::function<void(const void*)>> registered_buffer_;

public:
  /**
   * @brief Register GPU buffer using span
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the GPU buffer
   * @param flags Registration flags (default: 0)
   */
  template <typename T>
  explicit buffer_handle(cuda::std::span<T> buffer, int flags = 0);

  /**
   * @brief Register GPU buffer using span - const version
   * @tparam T Element type (must be trivially copyable)
   * @param buffer Span representing the GPU buffer
   * @param flags Registration flags (default: 0)
   */
  template <typename T>
  explicit buffer_handle(cuda::std::span<const T> buffer, int flags = 0);

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
  cuda::std::span<const cuda::std::byte> as_bytes() const noexcept;

  /**
   * @brief Get the buffer as a span of mutable bytes
   */
  cuda::std::span<cuda::std::byte> as_writable_bytes() const noexcept;

  /**
   * @brief Get the buffer as a typed span
   * @tparam T Element type (must be trivially copyable)
   * @return Span of type T over the buffer
   */
  template <typename T>
  cuda::std::span<T> as_span() const noexcept;

  /**
   * @brief Get the buffer as a typed const span
   * @tparam T Element type (must be trivially copyable)
   * @return Const span of type T over the buffer
   */
  template <typename T>
  cuda::std::span<const T> as_const_span() const noexcept;

  /**
   * @brief Check if the handle owns a valid resource
   */
  bool is_valid() const noexcept;
};

} // namespace cuda::experimental::cufile

#include "detail/buffer_handle_impl.hpp"
