//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_STREAM_POOL_H
#define _CUDA___STREAM_STREAM_POOL_H

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
#  include <cuda/__device/logical_device_ref.h>
#  include <cuda/__stream/invalid_stream.h>
#  include <cuda/__stream/relaxed_capture_scope.h>
#  include <cuda/__stream/stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>

#  include <mutex>
#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief A fixed-size pool of non-blocking streams on one device or green context.
//!
//! The pool owns its streams. Streams are created lazily on first use and destroyed with the pool.
//! `get_stream()` hands out the streams in round-robin order; `get_stream(i)` addresses slot `i % size()`.
//! Both return a `cuda::stream_ref` that stays valid for the lifetime of the pool, including across a
//! move of the pool itself.
class stream_pool
{
public:
  //! @brief Number of slots used when none is requested
  static constexpr ::cuda::std::size_t default_size = 16;

  //! @brief Constructs a pool of streams on the primary context of a device
  //!
  //! No stream is created until it is requested.
  //!
  //! @param[in] __device The device the streams are created on
  //! @param[in] __size Number of streams in the pool, must be greater than zero
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(
    device_ref __device, ::cuda::std::size_t __size = default_size, int __priority = stream::default_priority)
      : stream_pool{__logical_device_ref{__device}, __size, __priority}
  {}

  //! @brief Constructs a pool of streams on a logical device, that is a device or a green context
  //!
  //! No stream is created until it is requested. The pool does not own the green context, which must
  //! outlive the pool.
  //!
  //! @param[in] __device The logical device the streams are created on
  //! @param[in] __size Number of streams in the pool, must be greater than zero
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(
    __logical_device_ref __device, ::cuda::std::size_t __size = default_size, int __priority = stream::default_priority)
      : __device_{__device}
      , __priority_{__priority}
  {
    _CCCL_ASSERT(__size > 0, "cuda::stream_pool requires at least one stream");
    __streams_.reserve(__size);
    for (::cuda::std::size_t __i = 0; __i < __size; ++__i)
    {
      __streams_.emplace_back(no_init);
    }
  }

  stream_pool(const stream_pool&)            = delete;
  stream_pool& operator=(const stream_pool&) = delete;

  //! @brief Move-constructs a pool
  //!
  //! The streams and the round-robin position move over; the mutex of the new pool is a fresh one.
  //!
  //! @param[in,out] __other The pool to move from
  //!
  //! @post `__other` is empty and must not be used other than to be destroyed or assigned to
  _CCCL_HOST_API stream_pool(stream_pool&& __other) noexcept
      : __device_{__other.__device_}
      , __priority_{__other.__priority_}
      , __streams_{::cuda::std::move(__other.__streams_)}
      , __next_{::cuda::std::exchange(__other.__next_, 0)}
  {}

  //! @brief Move-assigns a pool
  //!
  //! The streams previously owned by this pool are destroyed.
  //!
  //! @param[in,out] __other The pool to move from
  //!
  //! @post `__other` is empty and must not be used other than to be destroyed or assigned to
  _CCCL_HOST_API stream_pool& operator=(stream_pool&& __other) noexcept
  {
    if (this != &__other)
    {
      __device_   = __other.__device_;
      __priority_ = __other.__priority_;
      __streams_  = ::cuda::std::move(__other.__streams_);
      __next_     = ::cuda::std::exchange(__other.__next_, 0);
    }
    return *this;
  }

  //! @brief Returns the next stream in round-robin order, creating it on first use
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref get_stream() const
  {
    const ::std::lock_guard<::std::mutex> __lock{__mutex_};
    _CCCL_ASSERT(!__streams_.empty(), "cuda::stream_pool::get_stream called on an empty pool");
    const ::cuda::std::size_t __index = __next_;
    __next_                           = (__next_ + 1) % __streams_.size();
    return __get_or_create(__index);
  }

  //! @brief Returns the stream in slot `__index % size()`, creating it on first use
  //!
  //! Requesting a slot does not advance the round-robin position.
  //!
  //! @param[in] __index Slot index, wraps around `size()`
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref get_stream(::cuda::std::size_t __index) const
  {
    const ::std::lock_guard<::std::mutex> __lock{__mutex_};
    _CCCL_ASSERT(!__streams_.empty(), "cuda::stream_pool::get_stream called on an empty pool");
    return __get_or_create(__index % __streams_.size());
  }

  //! @brief Number of streams in the pool
  //!
  //! Streams are created lazily, so this is the number of slots and not the number of streams
  //! created so far.
  //!
  //! @return The number of slots
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t size() const noexcept
  {
    return __streams_.size();
  }

  //! @brief The device the streams are created on
  //!
  //! @return The device, or the device owning the green context, given at construction
  [[nodiscard]] _CCCL_HOST_API device_ref device() const noexcept
  {
    return __device_.underlying_device();
  }

  //! @brief The logical device the streams are created on
  //!
  //! @return The logical device given at construction
  [[nodiscard]] _CCCL_HOST_API __logical_device_ref __logical_device() const noexcept
  {
    return __device_;
  }

  //! @brief The priority given to every stream in the pool
  //!
  //! @return The priority given at construction
  [[nodiscard]] _CCCL_HOST_API int priority() const noexcept
  {
    return __priority_;
  }

private:
  //! Returns the stream in slot `__index`, creating it if the slot is still empty. `__mutex_` must be held.
  _CCCL_HOST_API stream_ref __get_or_create(::cuda::std::size_t __index) const
  {
    stream& __slot = __streams_[__index];
    if (__slot.get() == ::cuda::__invalid_stream())
    {
      // Makes the stream creation capture-safe; a no-op when the calling thread is not capturing.
      const __relaxed_capture_scope __relaxed{};
      __slot = __create_stream();
    }
    return __slot;
  }

  //! Creates one stream on the logical device of the pool.
  [[nodiscard]] _CCCL_HOST_API stream __create_stream() const
  {
#  if _CCCL_CTK_AT_LEAST(12, 5)
    return stream{__device_, __priority_};
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv
    // Green contexts do not exist before CTK 12.5, so the logical device is always a plain device.
    return stream{__device_.underlying_device(), __priority_};
#  endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
  }

  __logical_device_ref __device_;
  int __priority_;
  mutable ::std::mutex __mutex_{};
  mutable ::std::vector<stream> __streams_{};
  mutable ::cuda::std::size_t __next_{0};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_STREAM_POOL_H
