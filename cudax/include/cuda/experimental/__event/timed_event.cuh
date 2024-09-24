//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_TIMED_EVENT_DETAIL_H
#define _CUDAX_TIMED_EVENT_DETAIL_H

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
#include <cuda/std/chrono>
#include <cuda/std/cstddef>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__event/event.cuh>

namespace cuda::experimental
{
//! @brief An owning wrapper for a `cudaEvent_t` with timing enabled.
class timed_event : public event
{
public:
  //! @brief Construct a new `timed_event` object with the specified flags
  //!        and record the event on the specified stream.
  //!
  //! @throws cuda_error if the event creation fails.
  explicit timed_event(stream_ref __stream, flags __flags = flags::none)
      : event(__stream, static_cast<unsigned int>(__flags))
  {
    record(__stream);
  }

  //! @brief Construct a new `timed_event` object into the moved-from state.
  //!
  //! @post `get()` returns `cudaEvent_t()`.
  explicit constexpr timed_event(uninit_t) noexcept
      : event(uninit)
  {}

  timed_event(timed_event&&) noexcept            = default;
  timed_event(const timed_event&)                = delete;
  timed_event& operator=(timed_event&&) noexcept = default;
  timed_event& operator=(const timed_event&)     = delete;

  //! @brief Construct a `timed_event` object from a native `cudaEvent_t` handle.
  //!
  //! @param __evnt The native handle
  //!
  //! @return timed_event The constructed `timed_event` object
  //!
  //! @note The constructed `timed_event` object takes ownership of the native handle.
  _CCCL_NODISCARD static timed_event from_native_handle(::cudaEvent_t __evnt) noexcept
  {
    return timed_event(__evnt);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static timed_event from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static timed_event from_native_handle(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Compute the time elapsed between two `timed_event` objects.
  //!
  //! @throws cuda_error if the query for the elapsed time fails.
  //!
  //! @param __end The `timed_event` object representing the end time.
  //! @param __start The `timed_event` object representing the start time.
  //!
  //! @return cuda::std::chrono::nanoseconds The elapsed time in nanoseconds.
  //!
  //! @note The elapsed time has a resolution of approximately 0.5 microseconds.
  _CCCL_NODISCARD_FRIEND _CUDA_VSTD::chrono::nanoseconds operator-(const timed_event& __end, const timed_event& __start)
  {
    float __ms = 0.0f;
    _CCCL_TRY_CUDA_API(
      ::cudaEventElapsedTime, "Failed to get CUDA event elapsed time", &__ms, __start.get(), __end.get());
    return _CUDA_VSTD::chrono::nanoseconds(static_cast<_CUDA_VSTD::chrono::nanoseconds::rep>(__ms * 1'000'000.0));
  }

private:
  // Use `timed_event::from_native_handle(e)` to construct an owning `timed_event`
  // object from a `cudaEvent_t` handle.
  explicit constexpr timed_event(::cudaEvent_t __evnt) noexcept
      : event(__evnt)
  {}
};
} // namespace cuda::experimental

#endif // _CUDAX_TIMED_EVENT_DETAIL_H
