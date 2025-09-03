//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___BARRIER_BARRIER_H
#define __CUDA_STD___BARRIER_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/poll_tester.h>
#include <cuda/std/__new_>
#include <cuda/std/atomic>
#include <cuda/std/chrono>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

template <class _CompletionF, thread_scope _Sco = thread_scope_system>
class __barrier_base
{
  __atomic_impl<ptrdiff_t, _Sco> __expected, __arrived;
  _CompletionF __completion;
  __atomic_impl<bool, _Sco> __phase;

public:
  using arrival_token = bool;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _CCCL_API inline friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _CCCL_API inline friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  [[nodiscard]] _CCCL_API inline bool __try_wait(arrival_token __old) const
  {
    return __phase.load(memory_order_acquire) != __old;
  }
  [[nodiscard]] _CCCL_API inline bool __try_wait_parity(bool __parity) const
  {
    return __try_wait(__parity);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _CCCL_API inline __barrier_base(ptrdiff_t __expected, _CompletionF __completion = _CompletionF())
      : __expected(__expected)
      , __arrived(__expected)
      , __completion(__completion)
      , __phase(false)
  {}

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  [[nodiscard]] _CCCL_API inline arrival_token arrive(ptrdiff_t __update = 1)
  {
    auto const __old_phase    = __phase.load(memory_order_relaxed);
    auto const __result       = __arrived.fetch_sub(__update, memory_order_acq_rel) - __update;
    auto const __new_expected = __expected.load(memory_order_relaxed);

    _CCCL_ASSERT(__result >= 0, "");

    if (0 == __result)
    {
      __completion();
      __arrived.store(__new_expected, memory_order_relaxed);
      __phase.store(!__old_phase, memory_order_release);
      __atomic_notify_all(&__phase.__a, __scope_to_tag<_Sco>{});
    }
    return __old_phase;
  }
  _CCCL_API inline void wait(arrival_token&& __old_phase) const
  {
    __phase.wait(__old_phase, memory_order_acquire);
  }
  _CCCL_API inline void arrive_and_wait()
  {
    wait(arrive());
  }
  _CCCL_API inline void arrive_and_drop()
  {
    __expected.fetch_sub(1, memory_order_relaxed);
    (void) arrive();
  }

  [[nodiscard]] _CCCL_API static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }
};

_CCCL_DIAG_POP

template <thread_scope _Sco>
class __barrier_base<__empty_completion, _Sco>
{
  static constexpr uint64_t __expected_unit = 1ull;
  static constexpr uint64_t __arrived_unit  = 1ull << 32;
  static constexpr uint64_t __expected_mask = __arrived_unit - 1;
  static constexpr uint64_t __phase_bit     = 1ull << 63;
  static constexpr uint64_t __arrived_mask  = (__phase_bit - 1) & ~__expected_mask;

  __atomic_impl<uint64_t, _Sco> __phase_arrived_expected;

public:
  using arrival_token = uint64_t;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _CCCL_API inline friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _CCCL_API inline friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  static _CCCL_API constexpr uint64_t __init(ptrdiff_t __count) noexcept
  {
    _CCCL_ASSERT(__count >= 0, "Count must be non-negative.");
    return (((1u << 31) - __count) << 32) | ((1u << 31) - __count);
  }
  [[nodiscard]] _CCCL_API inline bool __try_wait_phase(uint64_t __phase) const
  {
    uint64_t const __current = __phase_arrived_expected.load(memory_order_acquire);
    return ((__current & __phase_bit) != __phase);
  }
  [[nodiscard]] _CCCL_API inline bool __try_wait(arrival_token __old) const
  {
    return __try_wait_phase(__old & __phase_bit);
  }
  [[nodiscard]] _CCCL_API inline bool __try_wait_parity(bool __parity) const
  {
    return __try_wait_phase(__parity ? __phase_bit : 0);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _CCCL_API constexpr __barrier_base(ptrdiff_t __count, __empty_completion = __empty_completion())
      : __phase_arrived_expected(__init(__count))
  {
    _CCCL_ASSERT(__count >= 0, "");
  }

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  [[nodiscard]] _CCCL_API inline arrival_token arrive(ptrdiff_t __update = 1)
  {
    auto const __inc = __arrived_unit * __update;
    auto const __old = __phase_arrived_expected.fetch_add(__inc, memory_order_acq_rel);
    if ((__old ^ (__old + __inc)) & __phase_bit)
    {
      __phase_arrived_expected.fetch_add((__old & __expected_mask) << 32, memory_order_relaxed);
      __phase_arrived_expected.notify_all();
    }
    return __old & __phase_bit;
  }
  _CCCL_API inline void wait(arrival_token&& __phase) const
  {
    ::cuda::std::__cccl_thread_poll_with_backoff(
      __barrier_poll_tester_phase<__barrier_base>(this, ::cuda::std::move(__phase)));
  }
  _CCCL_API inline void wait_parity(bool __parity) const
  {
    ::cuda::std::__cccl_thread_poll_with_backoff(__barrier_poll_tester_parity<__barrier_base>(this, __parity));
  }
  _CCCL_API inline void arrive_and_wait()
  {
    wait(arrive());
  }
  _CCCL_API inline void arrive_and_drop()
  {
    __phase_arrived_expected.fetch_add(__expected_unit, memory_order_relaxed);
    (void) arrive();
  }

  [[nodiscard]] _CCCL_API static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<int32_t>::max();
  }
};

template <class _CompletionF = __empty_completion>
class barrier : public __barrier_base<_CompletionF>
{
public:
  _CCCL_API constexpr barrier(ptrdiff_t __count, _CompletionF __completion = _CompletionF())
      : __barrier_base<_CompletionF>(__count, __completion)
  {}
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___BARRIER_BARRIER_H
