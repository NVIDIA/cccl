//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EVENT_EVENT_H
#define _CUDA___EVENT_EVENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__event/event_ref.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__utility/no_init.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/cstddef>
#  include <cuda/std/utility>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

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
  explicit event(stream_ref __stream, flags __flags = flags::none);

  //! @brief Construct a new `event` object with timing disabled. The event can only be recorded on streams from the
  //! specified device.
  //!
  //! @throws cuda_error if the event creation fails.
  explicit event(device_ref __device, flags __flags = flags::none)
      : event(__device, static_cast<unsigned int>(__flags) | cudaEventDisableTiming)
  {}

  //! @brief Construct a new `event` object into the moved-from state.
  //!
  //! @post `get()` returns `cudaEvent_t()`.
  explicit constexpr event(no_init_t) noexcept
      : event_ref(::cudaEvent_t{})
  {}

  //! @brief Move-construct a new `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  constexpr event(event&& __other) noexcept
      : event_ref(::cuda::std::exchange(__other.__event_, {}))
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
      [[maybe_unused]] auto __status = ::cuda::__driver::__eventDestroyNoThrow(__event_);
    }
  }

  //! @brief Move-assign an `event` object
  //!
  //! @param __other
  //!
  //! @post `__other` is in a moved-from state.
  event& operator=(event&& __other) noexcept
  {
    event __tmp(::cuda::std::move(__other));
    ::cuda::std::swap(__event_, __tmp.__event_);
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
  [[nodiscard]] static event from_native_handle(::cudaEvent_t __evnt) noexcept
  {
    return event(__evnt);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static event from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static event from_native_handle(::cuda::std::nullptr_t) = delete;

  //! @brief Retrieve the native `cudaEvent_t` handle and give up ownership.
  //!
  //! @return cudaEvent_t The native handle being held by the `event` object.
  //!
  //! @post The event object is in a moved-from state.
  [[nodiscard]] constexpr ::cudaEvent_t release() noexcept
  {
    return ::cuda::std::exchange(__event_, {});
  }

  [[nodiscard]] friend constexpr flags operator|(flags __lhs, flags __rhs) noexcept
  {
    return static_cast<flags>(static_cast<unsigned int>(__lhs) | static_cast<unsigned int>(__rhs));
  }

private:
  // Use `event::from_native_handle(e)` to construct an owning `event`
  // object from a `cudaEvent_t` handle.
  explicit constexpr event(::cudaEvent_t __evnt) noexcept
      : event_ref(__evnt)
  {}

  explicit event(stream_ref __stream, unsigned int __flags);

  explicit event(device_ref __device, unsigned int __flags)
      : event_ref(::cudaEvent_t{})
  {
    [[maybe_unused]] __ensure_current_context __ctx_setter(__device);
    _CCCL_TRY_CUDA_API(
      ::cudaEventCreateWithFlags, "Failed to create CUDA event", &__event_, static_cast<unsigned int>(__flags));
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___EVENT_EVENT_H
