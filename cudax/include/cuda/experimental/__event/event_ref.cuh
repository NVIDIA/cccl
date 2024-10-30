//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_EVENT_REF_DETAIL_H
#define _CUDAX_EVENT_REF_DETAIL_H

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
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/__utility/driver_api.cuh>

namespace cuda::experimental
{
class event;
class timed_event;

//! @brief An non-owning wrapper for an untimed `cudaEvent_t`.
class event_ref
{
private:
  friend class event;
  friend class timed_event;

  ::cudaEvent_t __event_{};

public:
  using value_type = ::cudaEvent_t;

  //! @brief Construct a new `event_ref` object from a `cudaEvent_t`
  //!
  //! This constructor provides an implicit conversion from `cudaEvent_t`
  //!
  //! @post `get() == __evnt`
  //!
  //! @note: It is the callers responsibilty to ensure the `event_ref` does not
  //! outlive the event denoted by the `cudaEvent_t` handle.
  constexpr event_ref(::cudaEvent_t __evnt) noexcept
      : __event_(__evnt)
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  event_ref(int) = delete;

  /// Disallow construction from `nullptr`.
  event_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Records an event on the specified stream
  //!
  //! @param __stream
  //!
  //! @throws cuda_error if the event record fails
  void record(stream_ref __stream) const
  {
    _CCCL_ASSERT(__event_ != nullptr, "cuda::experimental::event_ref::record no event set");
    _CCCL_ASSERT(__stream.get() != nullptr, "cuda::experimental::event_ref::record invalid stream passed");
    // Need to use driver API, cudaEventRecord will push dev 0 if stack is empty
    detail::driver::eventRecord(__event_, __stream.get());
  }

  //! @brief Waits until all the work in the stream prior to the record of the
  //!        event has completed.
  //!
  //! @throws cuda_error if waiting for the event fails
  void wait() const
  {
    _CCCL_ASSERT(__event_ != nullptr, "cuda::experimental::event_ref::wait no event set");
    _CCCL_TRY_CUDA_API(::cudaEventSynchronize, "Failed to wait for CUDA event", __event_);
  }

  //! @brief Checks if all the work in the stream prior to the record of the event has completed.
  //!
  //! If is_done returns true, calling wait() on this event will return immediately
  //!
  //! @throws cuda_error if the event query fails
  _CCCL_NODISCARD bool is_done() const
  {
    _CCCL_ASSERT(__event_ != nullptr, "cuda::experimental::event_ref::wait no event set");
    cudaError_t __status = ::cudaEventQuery(__event_);
    if (__status == cudaSuccess)
    {
      return true;
    }
    else if (__status == cudaErrorNotReady)
    {
      return false;
    }
    else
    {
      ::cuda::__throw_cuda_error(__status, "Failed to query CUDA event");
    }
  }

  //! @brief Retrieve the native `cudaEvent_t` handle.
  //!
  //! @return cudaEvent_t The native handle being held by the event_ref object.
  _CCCL_NODISCARD constexpr ::cudaEvent_t get() const noexcept
  {
    return __event_;
  }

  //! @brief Checks if the `event_ref` is valid
  //!
  //! @return true if the `event_ref` is valid, false otherwise.
  _CCCL_NODISCARD explicit constexpr operator bool() const noexcept
  {
    return __event_ != nullptr;
  }

  //! @brief Compares two `event_ref`s for equality
  //!
  //! @note Allows comparison with `cudaEvent_t` due to implicit conversion to
  //! `event_ref`.
  //!
  //! @param __lhs The first `event_ref` to compare
  //! @param __rhs The second `event_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same `cudaEvent_t` object.
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ == __rhs.__event_;
  }

  //! @brief Compares two `event_ref`s for inequality
  //!
  //! @note Allows comparison with `cudaEvent_t` due to implicit conversion to
  //! `event_ref`.
  //!
  //! @param __lhs The first `event_ref` to compare
  //! @param __rhs The second `event_ref` to compare
  //! @return true if `lhs` and `rhs` refer to different `cudaEvent_t` objects.
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ != __rhs.__event_;
  }
};
} // namespace cuda::experimental

#endif // _CUDAX_EVENT_REF_DETAIL_H
