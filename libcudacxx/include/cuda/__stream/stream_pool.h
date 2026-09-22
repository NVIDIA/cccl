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
#  include <cuda/__stream/relaxed_capture_scope.h>
#  include <cuda/__stream/stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__atomic/order.h>
#  include <cuda/std/__atomic/platform.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__utility/exchange.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief When the streams of a `stream_pool` are created
enum class stream_pool_creation
{
  //! Every stream is created in the constructor
  eager,
  //! Each stream is created the first time its slot is requested
  lazy,
};

// Atomics on the round-robin counter and the slots of a stream_pool through the compiler builtins, so that the header
// does not pull in <atomic>. MSVC gets the same builtins from cuda/std/__atomic/platform.h, in namespace cuda::std.
#  if _CCCL_COMPILER(MSVC)
#    define _CUDA_STREAM_POOL_ATOMIC(__op) ::cuda::std::__op
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    define _CUDA_STREAM_POOL_ATOMIC(__op) __op
#  endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

//! @brief A fixed-size pool of non-blocking streams on one device or green context.
//!
//! The pool owns its streams and destroys them with the pool. `next_stream()` hands out the streams in
//! round-robin order; `at(i)` and `pool[i]` address slot `i % size()`. Both return a `cuda::stream_ref` that
//! stays valid for the lifetime of the pool. Destroying the pool destroys the streams; it is the caller's
//! responsibility to synchronize the work submitted to them first. The pool can be moved but not copied. A move
//! takes over the streams, which stay valid, as do the `cuda::stream_ref` handed out before the move; no thread may
//! use either pool while it is moved. A moved-from pool has a size of zero and may only be assigned to or destroyed.
//!
//! Whether the streams are created in the constructor or on the first request for their slot is chosen at
//! construction with a `stream_pool_creation` value. With `stream_pool_creation::eager`, the default, every stream
//! is created in the constructor. With `stream_pool_creation::lazy`, a stream is created by the first request for
//! its slot; two threads racing for the same empty slot both create a stream, one publishes it and the other
//! destroys its own. The getters can be called concurrently from several threads. The pool takes no lock: its
//! synchronization is lock-free, but not wait-free, including stream creation for lazily populated pools.
class stream_pool
{
public:
  //! @brief Constructs a pool of streams on the primary context of a device
  //!
  //! Every stream is created like a `cuda::stream`: non-blocking with respect to the legacy default stream, with
  //! the given priority.
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
  //! Every stream is created like a `cuda::stream`: non-blocking with respect to the legacy default stream, with
  //! the given priority. The pool does not own the green context, which must outlive the pool.
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
      , __size_{__size}
      , __slots_{__size == 0 ? nullptr : new ::cudaStream_t[__size]()}
  {
    if (__size == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "cuda::stream_pool requires at least one stream");
    }
    if (__mode == stream_pool_creation::eager)
    {
      _CCCL_TRY
      {
        // Makes the stream creation capture-safe; a no-op when the calling thread is not capturing.
        const __relaxed_capture_scope __relaxed{};
        for (::cuda::std::size_t __i = 0; __i < __size; ++__i)
        {
          // No other thread can see the pool yet, so a plain store suffices.
          __slots_[__i] = __create_stream().release();
        }
      }
      _CCCL_CATCH_ALL
      {
        __destroy_slots();
        _CCCL_RETHROW;
      }
    }
  }

  _CCCL_HOST_API ~stream_pool()
  {
    __destroy_slots();
  }

  stream_pool(const stream_pool&)            = delete;
  stream_pool& operator=(const stream_pool&) = delete;

  //! @brief Move-constructs a pool, taking over the streams of `__other`
  //!
  //! The streams, and the `cuda::stream_ref` handed out by `__other` before the move, stay valid. The round-robin
  //! position of `__other` is carried over. No thread may use `__other` during the move.
  //!
  //! @param[in,out] __other The pool to move from
  //!
  //! @post `__other` has a size of zero and may only be assigned to or destroyed
  _CCCL_HOST_API stream_pool(stream_pool&& __other) noexcept
      : __device_{__other.__device_}
      , __priority_{__other.__priority_}
      , __size_{::cuda::std::exchange(__other.__size_, ::cuda::std::size_t{0})}
      , __slots_{::cuda::std::exchange(__other.__slots_, nullptr)}
      , __next_{::cuda::std::exchange(__other.__next_, ::cuda::std::size_t{0})}
  {}

  //! @brief Move-assigns a pool, destroying the streams of this pool and taking over those of `__other`
  //!
  //! It is the caller's responsibility to synchronize the work submitted to the streams of this pool first. The
  //! streams of `__other`, and the `cuda::stream_ref` it handed out before the move, stay valid. No thread may use
  //! either pool during the move.
  //!
  //! @param[in,out] __other The pool to move from
  //!
  //! @return A reference to this pool
  //!
  //! @post `__other` has a size of zero and may only be assigned to or destroyed
  _CCCL_HOST_API stream_pool& operator=(stream_pool&& __other) noexcept
  {
    if (this != &__other)
    {
      __destroy_slots();
      __device_   = __other.__device_;
      __priority_ = __other.__priority_;
      __size_     = ::cuda::std::exchange(__other.__size_, ::cuda::std::size_t{0});
      __slots_    = ::cuda::std::exchange(__other.__slots_, nullptr);
      __next_     = ::cuda::std::exchange(__other.__next_, ::cuda::std::size_t{0});
    }
    return *this;
  }

  //! @brief Returns the next stream in round-robin order
  //!
  //! In a lazy pool, creates the stream if its slot is requested for the first time.
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref next_stream() const
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::stream_pool::next_stream called on a moved-from pool");
    // Advance the position and wrap it at size() in one compare-exchange, retried if another caller advanced it
    // in between. The position is always a valid slot, so the order is exact and nothing ever overflows.
    ::cuda::std::size_t __slot = __load_relaxed(&__next_);
    while (!__advance(&__next_, __slot, __slot + 1 == __size_ ? 0 : __slot + 1))
    {
    }
    return __stream_at(__slot);
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
  [[nodiscard]] _CCCL_HOST_API stream_ref at(::cuda::std::size_t __index) const
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::stream_pool::at called on a moved-from pool");
    return __stream_at(__index % __size_);
  }

  //! @brief Returns the stream in slot `__index % size()`, same as `at(__index)`
  //!
  //! @param[in] __index Slot index, wraps around `size()`
  //!
  //! @return A reference to a stream owned by the pool
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref operator[](::cuda::std::size_t __index) const
  {
    return at(__index);
  }

  //! @brief Number of streams in the pool
  //!
  //! Fixed at construction; every stream the pool ever hands out comes from one of these slots, whether or not
  //! its stream has been created yet.
  //!
  //! @return The size given at construction, zero for a moved-from pool
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t size() const noexcept
  {
    return __size_;
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
  //! @brief Relaxed atomic load of a round-robin position
  //!
  //! @param[in] __ptr The position to read
  //!
  //! @return The value stored at `__ptr`
  _CCCL_HOST_API static ::cuda::std::size_t __load_relaxed(const ::cuda::std::size_t* __ptr) noexcept
  {
    return _CUDA_STREAM_POOL_ATOMIC(__atomic_load_n)(__ptr, __ATOMIC_RELAXED);
  }

  //! @brief Weak relaxed compare-exchange advancing a round-robin position
  //!
  //! @param[in,out] __ptr The position to advance
  //! @param[in,out] __expected The value `__ptr` is expected to hold; on failure, set to the value it holds
  //! @param[in] __desired The value to store if `__ptr` holds `__expected`
  //!
  //! @return `true` if `__desired` was stored, `false` otherwise, including spuriously
  _CCCL_HOST_API static bool
  __advance(::cuda::std::size_t* __ptr, ::cuda::std::size_t& __expected, ::cuda::std::size_t __desired) noexcept
  {
    return _CUDA_STREAM_POOL_ATOMIC(
      __atomic_compare_exchange_n)(__ptr, &__expected, __desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  //! @brief Acquire atomic load of a slot
  //!
  //! @param[in] __ptr The slot to read
  //!
  //! @return The stream stored in the slot, `nullptr` if the slot is empty
  _CCCL_HOST_API static ::cudaStream_t __load_acquire(::cudaStream_t* __ptr) noexcept
  {
    return _CUDA_STREAM_POOL_ATOMIC(__atomic_load_n)(__ptr, __ATOMIC_ACQUIRE);
  }

  //! @brief Strong compare-exchange publishing a stream into an empty slot
  //!
  //! @param[in,out] __ptr The slot to fill
  //! @param[in,out] __expected The value the slot is expected to hold, `nullptr` for an empty slot; on failure, set to
  //! the stream another thread published
  //! @param[in] __desired The stream to publish
  //!
  //! @return `true` if `__desired` was published, `false` if the slot already held a stream
  _CCCL_HOST_API static bool
  __publish(::cudaStream_t* __ptr, ::cudaStream_t& __expected, ::cudaStream_t __desired) noexcept
  {
    return _CUDA_STREAM_POOL_ATOMIC(
      __atomic_compare_exchange_n)(__ptr, &__expected, __desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  }

  //! @brief Returns the stream of slot `__i`, creating it if the slot is still empty
  //!
  //! A slot changes exactly once, from empty to a stream that lives until the pool is destroyed, so a filled slot
  //! is read with a single acquire load. An empty slot is filled optimistically: the caller creates a stream and
  //! publishes it with a compare-exchange; if another thread published first, the caller destroys its own stream
  //! and returns the published one.
  //!
  //! @param[in] __i Slot index, must be below `size()`
  //!
  //! @return A reference to the stream of the slot
  //!
  //! @throws cuda_error if the stream has to be created and the creation fails
  [[nodiscard]] _CCCL_HOST_API stream_ref __stream_at(::cuda::std::size_t __i) const
  {
    ::cudaStream_t __published = __load_acquire(&__slots_[__i]);
    if (__published != nullptr)
    {
      return stream_ref{__published};
    }

    // Makes the stream creation, and its destruction if the publication loses, capture-safe; a no-op when the
    // calling thread is not capturing.
    const __relaxed_capture_scope __relaxed{};
    stream __fresh = __create_stream();
    if (__publish(&__slots_[__i], __published, __fresh.get()))
    {
      return stream_ref{__fresh.release()};
    }
    // Lost the race: `__fresh` is destroyed here, `__published` is what the winner stored.
    return stream_ref{__published};
  }

  //! @brief Destroys every published stream and frees the slots
  //!
  //! Called from the destructor, from the move assignment, and from the constructor when eager creation fails
  //! part-way.
  _CCCL_HOST_API void __destroy_slots() noexcept
  {
    if (__slots_ == nullptr)
    {
      return;
    }
    for (::cuda::std::size_t __i = 0; __i < __size_; ++__i)
    {
      if (__slots_[__i] != nullptr)
      {
        // Adopting the handle into a `stream` destroys it.
        const stream __owner = stream::from_native_handle(__slots_[__i]);
      }
    }
    delete[] __slots_;
    __slots_ = nullptr;
  }

  //! @brief Creates one stream on the logical device of the pool
  //!
  //! @return The new stream, non-blocking and with the priority of the pool
  //!
  //! @throws cuda_error if the stream creation fails
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
  //! Number of slots, zero only for a moved-from pool.
  ::cuda::std::size_t __size_;
  //! `__size_` slots; an empty slot holds `nullptr`, a filled slot the stream that lives until the pool is destroyed.
  //! Only ever accessed through the atomic helpers above, except in the constructors, the move
  //! assignment and `__destroy_slots()`, where no other thread can see the pool.
  ::cudaStream_t* __slots_;
  //! The slot the next call to `next_stream()` returns, always below `__size_`; only ever accessed through the
  //! atomic helpers above.
  mutable ::cuda::std::size_t __next_{0};
};

#  undef _CUDA_STREAM_POOL_ATOMIC

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_STREAM_POOL_H
