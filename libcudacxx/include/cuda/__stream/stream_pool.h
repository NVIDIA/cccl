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

#  include <atomic>
#  include <mutex>
#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief A fixed-size pool of non-blocking streams on one device or green context.
//!
//! The pool owns its streams and destroys them with the pool. `get_stream()` hands out the streams in
//! round-robin order; `get_stream(i)` addresses slot `i % capacity()`. Both return a `cuda::stream_ref` that
//! stays valid for the lifetime of the pool. Destroying the pool destroys the streams; it is the caller's
//! responsibility to synchronize the work submitted to them first. The pool can be neither copied nor moved; to
//! hand it around or share it, allocate it with `std::make_unique` or `std::make_shared`.
//!
//! By default, and with the `stream_pool::lazy` tag, a stream is created the first time its slot is requested,
//! and the getters take a mutex to do so. With the `stream_pool::eager` tag, every stream is created in the
//! constructor and the getters take no lock at all. All getters can be called concurrently from several threads.
class stream_pool
{
public:
  //! @brief Capacity used when none is requested
  static constexpr ::cuda::std::size_t default_capacity = 16;

  //! @brief Tag selecting the constructors that create each stream on the first request for its slot
  struct lazy_t
  {
    explicit lazy_t() = default;
  };

  //! @brief Tag selecting the constructors that create the streams in the constructor
  struct eager_t
  {
    explicit eager_t() = default;
  };

  //! @brief Tag value selecting lazy creation, see `lazy_t`; this is the default
  static constexpr lazy_t lazy{};

  //! @brief Tag value selecting eager creation, see `eager_t`
  static constexpr eager_t eager{};

  //! @brief Constructs a pool of streams on the primary context of a device
  //!
  //! Equivalent to the constructor taking `stream_pool::lazy`: a stream is created on the first request for
  //! its slot.
  //!
  //! @param[in] __device The device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(
    device_ref __device, ::cuda::std::size_t __capacity = default_capacity, int __priority = stream::default_priority)
      : stream_pool{lazy, __logical_device_ref{__device}, __capacity, __priority}
  {}

  //! @brief Constructs a pool of streams on a logical device, that is a device or a green context
  //!
  //! Equivalent to the constructor taking `stream_pool::lazy`: a stream is created on the first request for
  //! its slot. The pool does not own the green context, which must outlive the pool.
  //!
  //! @param[in] __device The logical device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(__logical_device_ref __device,
                                      ::cuda::std::size_t __capacity = default_capacity,
                                      int __priority                 = stream::default_priority)
      : stream_pool{lazy, __device, __capacity, __priority}
  {}

  //! @brief Constructs a pool of streams on the primary context of a device, creating each stream on demand
  //!
  //! No stream is created until its slot is requested. Nothing on the device is touched by the constructor.
  //!
  //! @param[in] __lazy Tag selecting lazy creation, pass `stream_pool::lazy`
  //! @param[in] __device The device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(
    lazy_t __lazy,
    device_ref __device,
    ::cuda::std::size_t __capacity = default_capacity,
    int __priority                 = stream::default_priority)
      : stream_pool{__lazy, __logical_device_ref{__device}, __capacity, __priority}
  {}

  //! @brief Constructs a pool of streams on a logical device, creating each stream on demand
  //!
  //! No stream is created until its slot is requested. Nothing on the device is touched by the constructor.
  //! The pool does not own the green context, which must outlive the pool.
  //!
  //! @param[in] __lazy Tag selecting lazy creation, pass `stream_pool::lazy`
  //! @param[in] __device The logical device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  _CCCL_HOST_API explicit stream_pool(
    lazy_t,
    __logical_device_ref __device,
    ::cuda::std::size_t __capacity = default_capacity,
    int __priority                 = stream::default_priority)
      : __device_{__device}
      , __priority_{__priority}
      , __lazy_{true}
  {
    _CCCL_ASSERT(__capacity > 0, "cuda::stream_pool requires at least one stream");
    __streams_.reserve(__capacity);
    for (::cuda::std::size_t __i = 0; __i < __capacity; ++__i)
    {
      __streams_.emplace_back(no_init);
    }
  }

  //! @brief Constructs a pool of streams on the primary context of a device and creates all of them
  //!
  //! @param[in] __eager Tag selecting eager creation, pass `stream_pool::eager`
  //! @param[in] __device The device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  //!
  //! @throws cuda_error if a stream creation fails
  _CCCL_HOST_API explicit stream_pool(
    eager_t __eager,
    device_ref __device,
    ::cuda::std::size_t __capacity = default_capacity,
    int __priority                 = stream::default_priority)
      : stream_pool{__eager, __logical_device_ref{__device}, __capacity, __priority}
  {}

  //! @brief Constructs a pool of streams on a logical device and creates all of them
  //!
  //! The pool does not own the green context, which must outlive the pool.
  //!
  //! @param[in] __eager Tag selecting eager creation, pass `stream_pool::eager`
  //! @param[in] __device The logical device the streams are created on
  //! @param[in] __capacity Number of stream slots, must be greater than zero, defaults to `default_capacity`
  //! @param[in] __priority Priority given to every stream, defaults to `stream::default_priority`
  //!
  //! @throws cuda_error if a stream creation fails
  _CCCL_HOST_API explicit stream_pool(
    eager_t,
    __logical_device_ref __device,
    ::cuda::std::size_t __capacity = default_capacity,
    int __priority                 = stream::default_priority)
      : __device_{__device}
      , __priority_{__priority}
      , __lazy_{false}
  {
    _CCCL_ASSERT(__capacity > 0, "cuda::stream_pool requires at least one stream");
    // Makes the stream creation capture-safe; a no-op when the calling thread is not capturing.
    const __relaxed_capture_scope __relaxed{};
    __streams_.reserve(__capacity);
    for (::cuda::std::size_t __i = 0; __i < __capacity; ++__i)
    {
      __streams_.emplace_back(__create_stream());
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
  [[nodiscard]] _CCCL_HOST_API stream_ref get_stream() const
  {
    // Wrapping around the counter only perturbs the order once every 2^64 requests.
    const ::cuda::std::size_t __ticket = __next_.fetch_add(1, ::std::memory_order_relaxed);
    return __stream_at(__ticket % __streams_.size());
  }

  //! @brief Returns the stream in slot `__index % capacity()`
  //!
  //! In a lazy pool, creates the stream if its slot is requested for the first time. Requesting a slot does
  //! not advance the round-robin position.
  //!
  //! @param[in] __index Slot index, wraps around `capacity()`
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref get_stream(::cuda::std::size_t __index) const
  {
    return __stream_at(__index % __streams_.size());
  }

  //! @brief Number of stream slots in the pool
  //!
  //! Fixed at construction; every stream the pool ever hands out comes from one of these slots, whether or not
  //! its stream has been created yet.
  //!
  //! @return The capacity given at construction
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t capacity() const noexcept
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
  //! Returns the stream of slot `__i`. An eager pool never changes its streams after construction, so no
  //! lock is needed to read one. A lazy pool takes the mutex and creates the stream on the first request.
  [[nodiscard]] _CCCL_HOST_API stream_ref __stream_at(::cuda::std::size_t __i) const
  {
    if (!__lazy_)
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
  bool __lazy_;
  //! Guards the creation of streams in a lazy pool. Unused in an eager pool.
  mutable ::std::mutex __mutex_{};
  //! The slots, `capacity()` of them; a slot without a stream holds `__invalid_stream()`.
  mutable ::std::vector<stream> __streams_{};
  mutable ::std::atomic<::cuda::std::size_t> __next_{0};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_STREAM_POOL_H
