//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___BARRIER_BARRIER_H
#define __LIBCUDACXX___BARRIER_BARRIER_H

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

#if _LIBCUDACXX_CUDA_ABI_VERSION < 3
#  define _LIBCUDACXX_BARRIER_ALIGNMENTS alignas(64)
#else // ^^^ _LIBCUDACXX_CUDA_ABI_VERSION < 3 ^^^ / vvv _LIBCUDACXX_CUDA_ABI_VERSION >= 3 vvv
#  define _LIBCUDACXX_BARRIER_ALIGNMENTS
#endif // _LIBCUDACXX_CUDA_ABI_VERSION >= 3

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CompletionF, thread_scope _Sco = thread_scope_system>
class __barrier_base
{
  _LIBCUDACXX_BARRIER_ALIGNMENTS __atomic_impl<ptrdiff_t, _Sco> __expected, __arrived;
  _LIBCUDACXX_BARRIER_ALIGNMENTS _CompletionF __completion;
  _LIBCUDACXX_BARRIER_ALIGNMENTS __atomic_impl<bool, _Sco> __phase;

public:
  using arrival_token = bool;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __old) const
  {
    return __phase.load(memory_order_acquire) != __old;
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __parity) const
  {
    return __try_wait(__parity);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _LIBCUDACXX_HIDE_FROM_ABI __barrier_base(ptrdiff_t __expected, _CompletionF __completion = _CompletionF())
      : __expected(__expected)
      , __arrived(__expected)
      , __completion(__completion)
      , __phase(false)
  {}

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI arrival_token arrive(ptrdiff_t __update = 1)
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
  _LIBCUDACXX_HIDE_FROM_ABI void wait(arrival_token&& __old_phase) const
  {
    __phase.wait(__old_phase, memory_order_acquire);
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait()
  {
    wait(arrive());
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_drop()
  {
    __expected.fetch_sub(1, memory_order_relaxed);
    (void) arrive();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<ptrdiff_t>::max();
  }
};

template <thread_scope _Sco>
class __barrier_base<__empty_completion, _Sco>
{
  static constexpr uint64_t __expected_unit = 1ull;
  static constexpr uint64_t __arrived_unit  = 1ull << 32;
  static constexpr uint64_t __expected_mask = __arrived_unit - 1;
  static constexpr uint64_t __phase_bit     = 1ull << 63;
  static constexpr uint64_t __arrived_mask  = (__phase_bit - 1) & ~__expected_mask;

  _LIBCUDACXX_BARRIER_ALIGNMENTS __atomic_impl<uint64_t, _Sco> __phase_arrived_expected;

public:
  using arrival_token = uint64_t;

private:
  template <typename _Barrier>
  friend class __barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class __barrier_poll_tester_parity;
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait(const _Barrier& __b, typename _Barrier::arrival_token&& __phase);
  template <typename _Barrier>
  _LIBCUDACXX_HIDE_FROM_ABI friend bool __call_try_wait_parity(const _Barrier& __b, bool __parity);

  static _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __init(ptrdiff_t __count) noexcept
  {
#if _CCCL_STD_VER >= 2014
    // This debug assert is not supported in C++11 due to resulting in a
    // multi-statement constexpr function.
    _CCCL_ASSERT(__count >= 0, "Count must be non-negative.");
#endif // _CCCL_STD_VER >= 2014
    return (((1u << 31) - __count) << 32) | ((1u << 31) - __count);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_phase(uint64_t __phase) const
  {
    uint64_t const __current = __phase_arrived_expected.load(memory_order_acquire);
    return ((__current & __phase_bit) != __phase);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __old) const
  {
    return __try_wait_phase(__old & __phase_bit);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __parity) const
  {
    return __try_wait_phase(__parity ? __phase_bit : 0);
  }

public:
  _CCCL_HIDE_FROM_ABI __barrier_base() = default;

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14
  __barrier_base(ptrdiff_t __count, __empty_completion = __empty_completion())
      : __phase_arrived_expected(__init(__count))
  {
    _CCCL_ASSERT(__count >= 0, "");
  }

  _CCCL_HIDE_FROM_ABI ~__barrier_base() = default;

  __barrier_base(__barrier_base const&)            = delete;
  __barrier_base& operator=(__barrier_base const&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI arrival_token arrive(ptrdiff_t __update = 1)
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
  _LIBCUDACXX_HIDE_FROM_ABI void wait(arrival_token&& __phase) const
  {
    __libcpp_thread_poll_with_backoff(__barrier_poll_tester_phase<__barrier_base>(this, _CUDA_VSTD::move(__phase)));
  }
  _LIBCUDACXX_HIDE_FROM_ABI void wait_parity(bool __parity) const
  {
    __libcpp_thread_poll_with_backoff(__barrier_poll_tester_parity<__barrier_base>(this, __parity));
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait()
  {
    wait(arrive());
  }
  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_drop()
  {
    __phase_arrived_expected.fetch_add(__expected_unit, memory_order_relaxed);
    (void) arrive();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return numeric_limits<int32_t>::max();
  }
};

template <class _CompletionF = __empty_completion>
class barrier : public __barrier_base<_CompletionF>
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr barrier(ptrdiff_t __count, _CompletionF __completion = _CompletionF())
      : __barrier_base<_CompletionF>(__count, __completion)
  {}
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___BARRIER_BARRIER_H
