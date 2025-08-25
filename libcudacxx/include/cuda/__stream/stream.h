//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_STREAM_H
#define _CUDA___STREAM_STREAM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__stream/stream_ref.h> // IWYU pragma: export
#  include <cuda/std/__cuda/api_wrapper.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief An owning wrapper for cudaStream_t.
struct stream : stream_ref
{
  // 0 is documented as default priority
  static constexpr int default_priority = 0;

  //! @brief Constructs a stream on a specified device and with specified priority
  //!
  //! Priority is defaulted to stream::default_priority
  //!
  //! @throws cuda_error if stream creation fails
  explicit stream(device_ref __dev, int __priority = default_priority)
      : stream_ref(__detail::__invalid_stream)
  {
    [[maybe_unused]] __ensure_current_context __ctx_setter(__dev);
    _CCCL_TRY_CUDA_API(
      ::cudaStreamCreateWithPriority, "Failed to create a stream", &__stream, cudaStreamNonBlocking, __priority);
  }

  //! @brief Construct a new `stream` object into the moved-from state.
  //!
  //! @post `stream()` returns an invalid stream handle
  // Can't be constexpr because __invalid_stream isn't
  explicit stream(no_init_t) noexcept
      : stream_ref(__detail::__invalid_stream)
  {}

  //! @brief Move-construct a new `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in moved-from state.
  stream(stream&& __other) noexcept
      : stream(::cuda::std::exchange(__other.__stream, __detail::__invalid_stream))
  {}

  stream(const stream&) = delete;

  //! Destroy the `stream` object
  //!
  //! @note If the stream fails to be destroyed, the error is silently ignored.
  ~stream()
  {
    if (__stream != __detail::__invalid_stream)
    {
      // Needs to call driver API in case current device is not set, runtime version would set dev 0 current
      // Alternative would be to store the device and push/pop here
      [[maybe_unused]] auto status = ::cuda::__driver::__streamDestroyNoThrow(__stream);
    }
  }

  //! @brief Move-assign a `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  stream& operator=(stream&& __other) noexcept
  {
    stream __tmp(::cuda::std::move(__other));
    ::cuda::std::swap(__stream, __tmp.__stream);
    return *this;
  }

  stream& operator=(const stream&) = delete;

  //! @brief Construct an `stream` object from a native `cudaStream_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return stream The constructed `stream` object
  //!
  //! @note The constructed `stream` object takes ownership of the native handle.
  [[nodiscard]] static stream from_native_handle(::cudaStream_t __handle)
  {
    return stream(__handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static stream from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static stream from_native_handle(::cuda::std::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaStream_t` handle and give up ownership.
  //!
  //! @return cudaStream_t The native handle being held by the `stream` object.
  //!
  //! @post The stream object is in a moved-from state.
  [[nodiscard]] ::cudaStream_t release()
  {
    return ::cuda::std::exchange(__stream, __detail::__invalid_stream);
  }

private:
  // Use `stream::from_native_handle(s)` to construct an owning `stream`
  // object from a `cudaStream_t` handle.
  explicit stream(::cudaStream_t __handle)
      : stream_ref(__handle)
  {}
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_STREAM_H
