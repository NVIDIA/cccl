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
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/cstddef>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cassert>

namespace cuda::experimental
{
class event;
class timed_event;

/**
 * @brief An non-owning wrapper for an untimed `cudaEvent_t`.
 */
class event_ref
{
private:
  friend class event;
  friend class timed_event;

  ::cudaEvent_t __event_{0};

public:
  using value_type = ::cudaEvent_t;

  /**
   * @brief Construct a new `event_ref` that does refer to an event.
   */
  event_ref() = default;

  /**
   * @brief Construct a new `event_ref` object from a `cudaEvent_t`
   *
   * This constructor provides an implicit conversion from `cudaEvent_t`
   *
   * @post `get() == __evnt`
   *
   * @note: It is the callers responsibilty to ensure the `event_ref` does not
   * outlive the event denoted by the `cudaEvent_t` handle.
   */
  constexpr event_ref(::cudaEvent_t __evnt) noexcept
      : __event_(__evnt)
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  event_ref(int) = delete;

  /// Disallow construction from `nullptr`.
  event_ref(_CUDA_VSTD::nullptr_t) = delete;

  /**
   * @brief Records an event on the specified stream
   *
   * @param __stream
   *
   * @throws cuda_error if the event record fails
   */
  void record(stream_ref __stream)
  {
    assert(__event_ != nullptr);
    assert(__stream.get() != nullptr);
    _CCCL_TRY_CUDA_API(::cudaEventRecord, "Failed to record CUDA event", __event_, __stream.get());
  }

  /**
   * @brief Waits for a CUDA event_ref to complete on the specified stream
   *
   * @param __stream The stream to wait on
   *
   * @throws cuda_error if the event_ref wait fails
   */
  void wait(stream_ref __stream) const
  {
    assert(__event_ != nullptr);
    assert(__stream.get() != nullptr);
    _CCCL_TRY_CUDA_API(::cudaStreamWaitEvent, "Failed to wait for CUDA event", __stream.get(), __event_);
  }

  /**
   * @brief Retrieve the native `cudaEvent_t` handle.
   *
   * @return cudaEvent_t The native handle being held by the event_ref object.
   */
  _CCCL_NODISCARD constexpr ::cudaEvent_t get() const noexcept
  {
    return __event_;
  }

  /**
   * @brief Compares two `event_ref`s for equality
   *
   * @note Allows comparison with `cudaEvent_t` due to implicit conversion to
   * `event_ref`.
   *
   * @param lhs The first `event_ref` to compare
   * @param rhs The second `event_ref` to compare
   * @return true if `lhs` and `rhs` refer to the same `cudaEvent_t` object.
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator==(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ == __rhs.__event_;
  }

  /**
   * @brief Compares two `event_ref`s for inequality
   *
   * @note Allows comparison with `cudaEvent_t` due to implicit conversion to
   * `event_ref`.
   *
   * @param lhs The first `event_ref` to compare
   * @param rhs The second `event_ref` to compare
   * @return true if `lhs` and `rhs` refer to different `cudaEvent_t` objects.
   */
  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(event_ref __lhs, event_ref __rhs) noexcept
  {
    return __lhs.__event_ != __rhs.__event_;
  }
};
} // namespace cuda::experimental

#endif // _CUDAX_EVENT_REF_DETAIL_H
