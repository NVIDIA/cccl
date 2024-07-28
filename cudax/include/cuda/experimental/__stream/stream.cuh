//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_STREAM
#define _CUDAX__STREAM_STREAM

#include <cuda_runtime_api.h>
// cuda_runtime_api needs to come first

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__device/device.cuh>
#include <cuda/experimental/__event/timed_event.cuh>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/utility>
#include <cuda/stream_ref>

namespace cuda::experimental
{

namespace detail
{
// 0 is a valid stream in CUDA, so we need some other invalid stream representation
// Can't make it constexpr, because cudaStream_t is a pointer type
static const cudaStream_t invalid_stream = reinterpret_cast<cudaStream_t>(~0ULL);
} // namespace detail

//! @brief An owning wrapper for cudaStream_t.
struct stream : stream_ref
{
  // 0 is documented as default priority
  static constexpr int default_priority = 0;

  //! @brief Constructs a stream on a specified device and with specified priority
  //!
  //! @throws cuda_error if stream creation fails
  explicit stream(device __dev, int __priority)
  {
    __scoped_device dev_setter(__dev);
    _CCCL_TRY_CUDA_API(
      ::cudaStreamCreateWithPriority, "Failed to create a stream", &__stream, cudaStreamDefault, __priority);
  }

  //! @brief Constructs a stream on a specified device and with default priority
  //!
  //! @throws cuda_error if stream creation fails
  explicit stream(device __dev)
      : stream(__dev, default_priority)
  {}

  //! @brief Constructs a stream on the default device
  //!
  //! @throws cuda_error if stream creation fails.
  stream()
      : stream(device{0})
  {}

  //! @brief Construct a new `stream` object into the moved-from state.
  //!
  //! @post `stream()` returns an invalid stream handle
  // Can't be constexpr because invalid_stream isn't
  explicit stream(uninit_t) noexcept
      : stream_ref(detail::invalid_stream)
  {}

  //! @brief Move-construct a new `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in moved-from state.
  stream(stream&& __other) noexcept
      : stream(_CUDA_VSTD::exchange(__other.__stream, detail::invalid_stream))
  {}

  // Disallow copy construction.
  stream(const stream&) = delete;

  //! Destroy the `stream` object
  //!
  //! @note If the stream fails to be destroyed, the error is silently ignored.
  ~stream()
  {
    if (__stream != detail::invalid_stream)
    {
      [[maybe_unused]] auto status = ::cudaStreamDestroy(__stream);
    }
  }

  //! @brief Move-assign a `stream` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  stream& operator=(stream&& __other)
  {
    stream __tmp(_CUDA_VSTD::move(__other));
    _CUDA_VSTD::swap(__stream, __tmp.__stream);
    return *this;
  }

  // Disallow copy assignment.
  stream& operator=(const stream&) = delete;

  // Ideally records and waits below would be in stream_ref, but we can't have it depend on cudax yet

  //! @brief Create a new event and record it into this stream
  //!
  //! @return A new event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD event record_event(event::flags __flags = event::flags::none) const
  {
    return event(*this, __flags);
  }

  //! @brief Create a new timed event and record it into this stream
  //!
  //! @return A new timed event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD timed_event record_timed_event(event::flags __flags = event::flags::none) const
  {
    return timed_event(*this, __flags);
  }

  using stream_ref::wait;

  //! @brief Make all future work submitted into this stream depend on completion of the specified event
  //!
  //! @param __ev Event that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  void wait(event_ref __ev) const
  {
    assert(__ev.get() != nullptr);
    _CCCL_TRY_CUDA_API(::cudaStreamWaitEvent, "Failed to make a stream wait for an event", get(), __ev.get());
  }

  //! @brief Make all future work submitted into this stream depend on completion of all work from the specified stream
  //!
  //! @param __other Stream that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  void wait(const stream_ref __other) const
  {
    // TODO consider an optimization to not create an event every time and instead have one persistent event or one per
    // stream
    assert(__stream.get() != nullptr);
    event __tmp(__other);
    wait(__tmp);
  }

  //! @brief Construct an `stream` object from a native `cudaStream_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return stream The constructed `stream` object
  //!
  //! @note The constructed `stream` object takes ownership of the native handle.
  _CCCL_NODISCARD static stream from_native_handle(cudaStream_t __handle)
  {
    return stream(__handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static stream from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static stream from_native_handle(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaStream_t` handle and give up ownership.
  //!
  //! @return cudaStream_t The native handle being held by the `stream` object.
  //!
  //! @post The stream object is in a moved-from state.
  _CCCL_NODISCARD cudaStream_t release()
  {
    return _CUDA_VSTD::exchange(__stream, detail::invalid_stream);
  }

private:
  // Use `stream::from_native_handle(s)` to construct an owning `stream`
  // object from a `cudaStream_t` handle.
  explicit stream(cudaStream_t __handle)
      : stream_ref(__handle)
  {}
};

} // namespace cuda::experimental

#endif
