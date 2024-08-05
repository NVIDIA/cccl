//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_EVENT_DETAIL_H
#define _CUDAX_EVENT_DETAIL_H

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

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__event/event_ref.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

namespace cuda::experimental
{
class timed_event;

//! @brief An owning wrapper for an untimed `cudaEvent_t`.
class event : public event_ref
{
  friend class timed_event;

public:
  //! @brief Flags to use when creating the event.
  enum class flags : unsigned int
  {
    none          = cudaEventDefault,
    blocking_sync = cudaEventBlockingSync,
    interprocess  = cudaEventInterprocess
  };

  //! @brief Construct a new `event` object with timing disabled, and record
  //!        the event in the specified stream.
  //!
  //! @throws cuda_error if the event creation fails.
  explicit event(stream_ref __stream, flags __flags = flags::none)
      : event(__stream, static_cast<unsigned int>(__flags) | cudaEventDisableTiming)
  {
    record(__stream);
  }

  //! @brief Construct a new `event` object into the moved-from state.
  //!
  //! @post `get()` returns `cudaEvent_t()`.
  explicit constexpr event(uninit_t) noexcept
      : event_ref(::cudaEvent_t{})
  {}

  //! @brief Move-construct a new `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  constexpr event(event&& __other) noexcept
      : event_ref(_CUDA_VSTD::exchange(__other.__event_, {}))
  {}

  // Disallow copy construction.
  event(const event&) = delete;

  //! @brief Destroy the `event` object
  //!
  //! @note If the event fails to be destroyed, the error is silently ignored.
  ~event()
  {
    if (__event_ != nullptr)
    {
      // Needs to call driver API in case current device is not set, runtime version would set dev 0 current
      // Alternative would be to store the device and push/pop here
      [[maybe_unused]] auto __status = detail::driver::eventDestroy(__event_);
    }
  }

  //! @brief Move-assign an `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  event& operator=(event&& __other) noexcept
  {
    event __tmp(_CUDA_VSTD::move(__other));
    _CUDA_VSTD::swap(__event_, __tmp.__event_);
    return *this;
  }

  // Disallow copy assignment.
  event& operator=(const event&) = delete;

  //! @brief Construct an `event` object from a native `cudaEvent_t` handle.
  //!
  //! @param __evnt The native handle
  //!
  //! @return event The constructed `event` object
  //!
  //! @note The constructed `event` object takes ownership of the native handle.
  _CCCL_NODISCARD static event from_native_handle(::cudaEvent_t __evnt) noexcept
  {
    return event(__evnt);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static event from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static event from_native_handle(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaEvent_t` handle and give up ownership.
  //!
  //! @return cudaEvent_t The native handle being held by the `event` object.
  //!
  //! @post The event object is in a moved-from state.
  _CCCL_NODISCARD constexpr ::cudaEvent_t release() noexcept
  {
    return _CUDA_VSTD::exchange(__event_, {});
  }

  _CCCL_NODISCARD_FRIEND constexpr flags operator|(flags __lhs, flags __rhs) noexcept
  {
    return static_cast<flags>(static_cast<unsigned int>(__lhs) | static_cast<unsigned int>(__rhs));
  }

private:
  // Use `event::from_native_handle(e)` to construct an owning `event`
  // object from a `cudaEvent_t` handle.
  explicit constexpr event(::cudaEvent_t __evnt) noexcept
      : event_ref(__evnt)
  {}

  explicit event(stream_ref __stream, unsigned int __flags)
      : event_ref(::cudaEvent_t{})
  {
    [[maybe_unused]] __ensure_current_device __dev_setter(__stream);
    _CCCL_TRY_CUDA_API(
      ::cudaEventCreateWithFlags, "Failed to create CUDA event", &__event_, static_cast<unsigned int>(__flags));
  }
};
} // namespace cuda::experimental

#endif // _CUDAX_EVENT_DETAIL_H
