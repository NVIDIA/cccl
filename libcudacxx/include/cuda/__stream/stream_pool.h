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
#  include <cuda/std/__atomic/order.h>
#  include <cuda/std/__atomic/platform.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__limits/numeric_limits.h>

#  include <mutex>
#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Relaxed atomics on the round-robin counter of a stream_pool through the compiler builtins, so that the header does
// not pull in <atomic>. MSVC gets the same builtins from cuda/std/__atomic/platform.h, in namespace cuda::std.
#  if _CCCL_COMPILER(MSVC)
#    define _CUDA_STREAM_POOL_ATOMIC(__op) ::cuda::std::__op
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    define _CUDA_STREAM_POOL_ATOMIC(__op) __op
#  endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

_CCCL_HOST_API inline ::cuda::std::size_t
__stream_pool_fetch_add_relaxed(::cuda::std::size_t* __ptr, ::cuda::std::size_t __val) noexcept
{
  return _CUDA_STREAM_POOL_ATOMIC(__atomic_fetch_add)(__ptr, __val, __ATOMIC_RELAXED);
}

_CCCL_HOST_API inline ::cuda::std::size_t
__stream_pool_fetch_sub_relaxed(::cuda::std::size_t* __ptr, ::cuda::std::size_t __val) noexcept
{
  return _CUDA_STREAM_POOL_ATOMIC(__atomic_fetch_sub)(__ptr, __val, __ATOMIC_RELAXED);
}

_CCCL_HOST_API inline ::cuda::std::size_t __stream_pool_load_relaxed(const ::cuda::std::size_t* __ptr) noexcept
{
  return _CUDA_STREAM_POOL_ATOMIC(__atomic_load_n)(__ptr, __ATOMIC_RELAXED);
}

_CCCL_HOST_API inline void __stream_pool_store_relaxed(::cuda::std::size_t* __ptr, ::cuda::std::size_t __val) noexcept
{
  _CUDA_STREAM_POOL_ATOMIC(__atomic_store_n)(__ptr, __val, __ATOMIC_RELAXED);
}

#  undef _CUDA_STREAM_POOL_ATOMIC

//! @brief When the streams of a `stream_pool` are created
enum class stream_pool_creation
{
  //! Every stream is created in the constructor
  eager,
  //! Each stream is created the first time its slot is requested
  lazy,
};

//! @brief A fixed-size pool of non-blocking streams on one device or green context.
//!
//! The pool owns its streams and destroys them with the pool. `next_stream()` hands out the streams in
//! round-robin order; `get_stream(i)` addresses slot `i % size()`. Both return a `cuda::stream_ref` that
//! stays valid for the lifetime of the pool. Destroying the pool destroys the streams; it is the caller's
//! responsibility to synchronize the work submitted to them first. The pool can be neither copied nor moved; to
//! hand it around or share it, allocate it with `std::make_unique` or `std::make_shared`.
//!
//! Whether the streams are created in the constructor or on the first request for their slot is chosen at
//! construction with a `stream_pool_creation` value. With `stream_pool_creation::eager`, the default, every stream
//! is created in the constructor and the getters take no lock at all. With `stream_pool_creation::lazy`, the getters
//! take a mutex to create a stream the first time its slot is requested. All getters can be called concurrently from
//! several threads.
class stream_pool
{
public:
  //! @brief Constructs a pool of streams on the primary context of a device
  //!
  //! @param[in] __device The device the streams are created on
  //! @param[in] __size Number of streams in the pool, must be greater than zero
  //! @param[in] __mode When the streams are created, defaults to `stream_pool_creation::eager`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  //!
  //! @throws std::invalid_argument if `__size` is zero
  //! @throws cuda_error if `__mode` is `stream_pool_creation::eager` and a stream creation fails
  _CCCL_HOST_API explicit stream_pool(
    device_ref __device,
    ::cuda::std::size_t __size,
    stream_pool_creation __mode = stream_pool_creation::eager,
    int __priority              = stream::default_priority)
      : stream_pool{__logical_device_ref{__device}, __size, __mode, __priority}
  {}

  //! @brief Constructs a pool of streams on a logical device, that is a device or a green context
  //!
  //! The pool does not own the green context, which must outlive the pool.
  //!
  //! @param[in] __device The logical device the streams are created on
  //! @param[in] __size Number of streams in the pool, must be greater than zero
  //! @param[in] __mode When the streams are created, defaults to `stream_pool_creation::eager`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  //!
  //! @throws std::invalid_argument if `__size` is zero
  //! @throws cuda_error if `__mode` is `stream_pool_creation::eager` and a stream creation fails
  _CCCL_HOST_API explicit stream_pool(
    __logical_device_ref __device,
    ::cuda::std::size_t __size,
    stream_pool_creation __mode = stream_pool_creation::eager,
    int __priority              = stream::default_priority)
      : __device_{__device}
      , __priority_{__priority}
      , __mode_{__mode}
      , __wrap_{__wrap_ticket_for(__size)}
  {
    if (__size == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "cuda::stream_pool requires at least one stream");
    }
    __streams_.reserve(__size);
    if (__mode_ == stream_pool_creation::lazy)
    {
      for (::cuda::std::size_t __i = 0; __i < __size; ++__i)
      {
        __streams_.emplace_back(no_init);
      }
    }
    else
    {
      // Makes the stream creation capture-safe; a no-op when the calling thread is not capturing.
      const __relaxed_capture_scope __relaxed{};
      for (::cuda::std::size_t __i = 0; __i < __size; ++__i)
      {
        __streams_.emplace_back(__create_stream());
      }
    }
  }

  stream_pool(const stream_pool&)            = delete;
  stream_pool& operator=(const stream_pool&) = delete;

  stream_pool(stream_pool&&)            = delete;
  stream_pool& operator=(stream_pool&&) = delete;

  //! @brief Returns the next stream in round-robin order
  //!
  //! In a lazy pool, creates the stream if its slot is requested for the first time.
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref next_stream() const
  {
    const ::cuda::std::size_t __ticket = ::cuda::__stream_pool_fetch_add_relaxed(&__next_, 1);
    if (__ticket == __wrap_)
    {
      // Tickets are unique, so exactly one caller draws `__wrap_` and it alone pulls the counter back. Until
      // its subtraction lands, other callers keep drawing `__wrap_ + 1`, `__wrap_ + 2`, ...: that is fine,
      // `__wrap_` is a multiple of `size()`, so the modulo below maps those tickets to slots 1, 2, ..., exactly
      // the slots that follow the wrap ticket. Once the subtraction lands the counter continues from the same
      // slot sequence, so the round-robin order is exact and the counter never overflows.
      ::cuda::__stream_pool_fetch_sub_relaxed(&__next_, __wrap_);
    }
    return __stream_at(__ticket % __streams_.size());
  }

  //! @brief Returns the stream in slot `__index % size()`
  //!
  //! In a lazy pool, creates the stream if its slot is requested for the first time. Requesting a slot does
  //! not advance the round-robin position.
  //!
  //! @param[in] __index Slot index, wraps around `size()`
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref get_stream(::cuda::std::size_t __index) const
  {
    return __stream_at(__index % __streams_.size());
  }

  //! @brief Number of streams in the pool
  //!
  //! Fixed at construction; every stream the pool ever hands out comes from one of these slots, whether or not
  //! its stream has been created yet.
  //!
  //! @return The size given at construction
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

  //! @brief The round-robin ticket the next call to `next_stream()` draws. For tests only.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __next_ticket() const noexcept
  {
    return ::cuda::__stream_pool_load_relaxed(&__next_);
  }

  //! @brief Sets the round-robin ticket the next call to `next_stream()` draws. For tests only.
  //!
  //! @param[in] __ticket The ticket, must not exceed `__wrap_ticket()`
  _CCCL_HOST_API void __set_next_ticket(::cuda::std::size_t __ticket) const noexcept
  {
    ::cuda::__stream_pool_store_relaxed(&__next_, __ticket);
  }

  //! @brief The ticket at which the round-robin counter is pulled back by that same amount. For tests only.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __wrap_ticket() const noexcept
  {
    return __wrap_;
  }

  //! @brief The priority given to every stream in the pool
  //!
  //! @return The priority given at construction
  [[nodiscard]] _CCCL_HOST_API int priority() const noexcept
  {
    return __priority_;
  }

private:
  //! Returns the stream of slot `__i`. An eager pool never changes its streams after construction, so no
  //! lock is needed to read one. A lazy pool takes the mutex and creates the stream on the first request.
  [[nodiscard]] _CCCL_HOST_API stream_ref __stream_at(::cuda::std::size_t __i) const
  {
    if (__mode_ == stream_pool_creation::eager)
    {
      return __streams_[__i];
    }
    const ::std::lock_guard<::std::mutex> __lock{__mutex_};
    stream& __slot = __streams_[__i];
    if (__slot.get() == ::cuda::__invalid_stream())
    {
      // Makes the stream creation capture-safe; a no-op when the calling thread is not capturing.
      const __relaxed_capture_scope __relaxed{};
      __slot = __create_stream();
    }
    return __slot;
  }

  //! The largest multiple of `__size` not above half the counter range: far enough that the counter cannot
  //! overflow before the caller drawing it has subtracted it, and a multiple of `__size` so the subtraction
  //! preserves every ticket's slot.
  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
  __wrap_ticket_for(::cuda::std::size_t __size) noexcept
  {
    constexpr ::cuda::std::size_t __half = ::cuda::std::numeric_limits<::cuda::std::size_t>::max() / 2;
    return __size == 0 ? __half : __half - __half % __size;
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

  const __logical_device_ref __device_;
  const int __priority_;
  const stream_pool_creation __mode_;
  //! The round-robin ticket at which the counter is pulled back by `__wrap_`; see `next_stream()`.
  const ::cuda::std::size_t __wrap_;
  //! Guards the creation of streams in a lazy pool. Unused in an eager pool.
  mutable ::std::mutex __mutex_{};
  //! The slots, `size()` of them; a slot without a stream holds `__invalid_stream()`.
  mutable ::std::vector<stream> __streams_{};
  //! The round-robin counter; only ever accessed through the relaxed atomic builtins at the top of this file.
  mutable ::cuda::std::size_t __next_{0};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_STREAM_POOL_H
