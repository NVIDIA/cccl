//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_BARRIER_BLOCK_SCOPE_H
#define _CUDA___BARRIER_BARRIER_BLOCK_SCOPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/barrier.h>
#include <cuda/__fwd/barrier_native_handle.h>
#if defined(_CCCL_CUDA_COMPILER)
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILER
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/poll_tester.h>
#include <cuda/std/__new_>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>

#include <nv/target>

#if defined(_CCCL_COMPILER_NVRTC)
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !(&(((type*) 0)->member))
#else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv !_CCCL_COMPILER_NVRTC vvv
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !offsetof(type, member)
#endif // _CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// Needed for pipeline.arrive_on
struct __block_scope_barrier_base
{};

template <>
class barrier<thread_scope_block, _CUDA_VSTD::__empty_completion> : public __block_scope_barrier_base
{
  using __barrier_base = _CUDA_VSTD::__barrier_base<_CUDA_VSTD::__empty_completion, thread_scope_block>;
  __barrier_base __barrier;

  _CCCL_DEVICE friend inline _CUDA_VSTD::uint64_t*
  device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(barrier<thread_scope_block>& b);

  template <typename _Barrier>
  friend class _CUDA_VSTD::__barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class _CUDA_VSTD::__barrier_poll_tester_parity;

public:
  using arrival_token           = typename __barrier_base::arrival_token;
  _CCCL_HIDE_FROM_ABI barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI barrier(_CUDA_VSTD::ptrdiff_t __expected,
                                    _CUDA_VSTD::__empty_completion __completion = _CUDA_VSTD::__empty_completion())
  {
    static_assert(_LIBCUDACXX_OFFSET_IS_ZERO(barrier<thread_scope_block>, __barrier),
                  "fatal error: bad barrier layout");
    init(this, __expected, __completion);
  }

  _LIBCUDACXX_HIDE_FROM_ABI ~barrier()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (__isShared(&__barrier)) {
          asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<_CUDA_VSTD::uint32_t>(
            __cvta_generic_to_shared(&__barrier)))
                       : "memory");
        } else if (__isClusterShared(&__barrier)) { __trap(); }),
      NV_PROVIDES_SM_80,
      (if (__isShared(&__barrier)) {
        asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<_CUDA_VSTD::uint32_t>(
          __cvta_generic_to_shared(&__barrier)))
                     : "memory");
      }))
  }

  _LIBCUDACXX_HIDE_FROM_ABI friend void init(
    barrier* __b, _CUDA_VSTD::ptrdiff_t __expected, _CUDA_VSTD::__empty_completion = _CUDA_VSTD::__empty_completion())
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (__isShared(&__b->__barrier)) {
          asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(
                         static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__b->__barrier))),
                       "r"(static_cast<_CUDA_VSTD::uint32_t>(__expected))
                       : "memory");
        } else if (__isClusterShared(&__b->__barrier)) { __trap(); } else {
          new (&__b->__barrier) __barrier_base(__expected);
        }),
      NV_PROVIDES_SM_80,
      (
        if (__isShared(&__b->__barrier)) {
          asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(
                         static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__b->__barrier))),
                       "r"(static_cast<_CUDA_VSTD::uint32_t>(__expected))
                       : "memory");
        } else { new (&__b->__barrier) __barrier_base(__expected); }),
      NV_ANY_TARGET,
      (new (&__b->__barrier) __barrier_base(__expected);))
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI arrival_token arrive(_CUDA_VSTD::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update >= 0, "Arrival count update must be non-negative.");
    arrival_token __token = {};
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!__isClusterShared(&__barrier)) { return __barrier.arrive(__update); } else if (!__isShared(&__barrier)) {
          __trap();
        }
        // Cannot use cuda::device::barrier_native_handle here, as it is
        // only defined for block-scope barriers. This barrier may be a
        // non-block scoped barrier.
        auto __bh = reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__barrier);
        __token   = _CUDA_VPTX::mbarrier_arrive(__bh, __update);),
      NV_PROVIDES_SM_80,
      (
        if (!__isShared(&__barrier)) {
          return __barrier.arrive(__update);
        } auto __bh = reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__barrier);
        // Need 2 instructions, can't finish barrier with arrive > 1
        if (__update > 1) { _CUDA_VPTX::mbarrier_arrive_no_complete(__bh, __update - 1); } __token =
          _CUDA_VPTX::mbarrier_arrive(__bh);),
      NV_PROVIDES_SM_70,
      (
        if (!__isShared(&__barrier)) { return __barrier.arrive(__update); }

        unsigned int __mask    = __activemask();
        unsigned int __activeA = __match_any_sync(__mask, __update);
        unsigned int __activeB = __match_any_sync(__mask, reinterpret_cast<_CUDA_VSTD::uintptr_t>(&__barrier));
        unsigned int __active  = __activeA & __activeB;
        int __inc              = __popc(__active) * __update;

        unsigned __laneid;
        asm("mov.u32 %0, %%laneid;"
            : "=r"(__laneid));
        int __leader = __ffs(__active) - 1;
        // All threads in mask synchronize here, establishing cummulativity to the __leader:
        __syncwarp(__mask);
        if (__leader == static_cast<int>(__laneid)) {
          __token = __barrier.arrive(__inc);
        } __token = __shfl_sync(__active, __token, __leader);),
      NV_IS_HOST,
      (__token = __barrier.arrive(__update);))
    return __token;
  }

private:
  _LIBCUDACXX_HIDE_FROM_ABI bool __test_wait_sm_80(arrival_token __token) const
  {
    (void) __token;
    int32_t __ready = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_80,
      (asm volatile("{\n\t"
                    ".reg .pred p;\n\t"
                    "mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
                    "selp.b32 %0, 1, 0, p;\n\t"
                    "}"
                    : "=r"(__ready)
                    : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))), "l"(__token)
                    : "memory");))
    return __ready;
  }

  // Document de drop > uint32_t for __nanosec on public for APIs
  _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __token) const
  {
    (void) __token;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0; if (!__isClusterShared(&__barrier)) {
          return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));
        } else if (!__isShared(&__barrier)) {
          __trap();
        } asm volatile("{\n\t"
                       ".reg .pred p;\n\t"
                       "mbarrier.try_wait.shared.b64 p, [%1], %2;\n\t"
                       "selp.b32 %0, 1, 0, p;\n\t"
                       "}"
                       : "=r"(__ready)
                       : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))), "l"(__token)
                       : "memory");
        return __ready;),
      NV_PROVIDES_SM_80,
      (if (!__isShared(&__barrier)) {
        return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));
      } return __test_wait_sm_80(__token);),
      NV_ANY_TARGET,
      (return _CUDA_VSTD::__call_try_wait(__barrier, _CUDA_VSTD::move(__token));))
  }

  // Document de drop > uint32_t for __nanosec on public for APIs
  _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait(arrival_token __token, _CUDA_VSTD::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait(_CUDA_VSTD::move(__token));
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0;
        if (!__isClusterShared(&__barrier)) {
          return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
            _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)), __nanosec);
        } else if (!__isShared(&__barrier)) { __trap(); }

        _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
          _CUDA_VSTD::chrono::high_resolution_clock::now();
        _CUDA_VSTD::chrono::nanoseconds __elapsed;
        do {
          const _CUDA_VSTD::uint32_t __wait_nsec = static_cast<_CUDA_VSTD::uint32_t>((__nanosec - __elapsed).count());
          asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(__ready)
            : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
              "l"(__token),
              "r"(__wait_nsec)
            : "memory");
          __elapsed = _CUDA_VSTD::chrono::high_resolution_clock::now() - __start;
        } while (!__ready && (__nanosec > __elapsed));
        return __ready;),
      NV_PROVIDES_SM_80,
      (
        bool __ready = 0;
        if (!__isShared(&__barrier)) {
          return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
            _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)), __nanosec);
        }

        _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
          _CUDA_VSTD::chrono::high_resolution_clock::now();
        do {
          __ready = __test_wait_sm_80(__token);
        } while (!__ready && __nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));
        return __ready;),
      NV_ANY_TARGET,
      (return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__token)),
                _CUDA_VSTD::chrono::nanoseconds(__nanosec));))
  }

  _LIBCUDACXX_HIDE_FROM_ABI bool __test_wait_parity_sm_80(bool __phase_parity) const
  {
    (void) __phase_parity;
    uint16_t __ready = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_80,
      (asm volatile(
         "{"
         ".reg .pred %%p;"
         "mbarrier.test_wait.parity.shared.b64 %%p, [%1], %2;"
         "selp.u16 %0, 1, 0, %%p;"
         "}"
         : "=h"(__ready)
         : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&__barrier))), "r"(static_cast<uint32_t>(__phase_parity))
         : "memory");))
    return __ready;
  }

  _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __phase_parity) const
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!__isClusterShared(&__barrier)) {
          return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity);
        } else if (!__isShared(&__barrier)) { __trap(); } int32_t __ready = 0;

        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
          "selp.b32 %0, 1, 0, p;\n\t"
          "}"
          : "=r"(__ready)
          : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
            "r"(static_cast<_CUDA_VSTD::uint32_t>(__phase_parity))
          :);

        return __ready;),
      NV_PROVIDES_SM_80,
      (if (!__isShared(&__barrier)) { return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity); }

       return __test_wait_parity_sm_80(__phase_parity);),
      NV_ANY_TARGET,
      (return _CUDA_VSTD::__call_try_wait_parity(__barrier, __phase_parity);))
  }

  _LIBCUDACXX_HIDE_FROM_ABI bool __try_wait_parity(bool __phase_parity, _CUDA_VSTD::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait_parity(__phase_parity);
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0;
        if (!__isClusterShared(&__barrier)) {
          return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
            _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
        } else if (!__isShared(&__barrier)) { __trap(); }

        _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
          _CUDA_VSTD::chrono::high_resolution_clock::now();
        _CUDA_VSTD::chrono::nanoseconds __elapsed;
        do {
          const _CUDA_VSTD::uint32_t __wait_nsec = static_cast<_CUDA_VSTD::uint32_t>((__nanosec - __elapsed).count());
          asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2, %3;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(__ready)
            : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(&__barrier))),
              "r"(static_cast<_CUDA_VSTD::uint32_t>(__phase_parity)),
              "r"(__wait_nsec)
            : "memory");
          __elapsed = _CUDA_VSTD::chrono::high_resolution_clock::now() - __start;
        } while (!__ready && (__nanosec > __elapsed));

        return __ready;),
      NV_PROVIDES_SM_80,
      (
        bool __ready = 0;
        if (!__isShared(&__barrier)) {
          return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
            _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
        }

        _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
          _CUDA_VSTD::chrono::high_resolution_clock::now();
        do {
          __ready = __test_wait_parity_sm_80(__phase_parity);
        } while (!__ready && __nanosec > (_CUDA_VSTD::chrono::high_resolution_clock::now() - __start));

        return __ready;),
      NV_ANY_TARGET,
      (return _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
                _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);))
  }

public:
  _LIBCUDACXX_HIDE_FROM_ABI void wait(arrival_token&& __phase) const
  {
    _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
      _CUDA_VSTD::__barrier_poll_tester_phase<barrier>(this, _CUDA_VSTD::move(__phase)));
  }

  _LIBCUDACXX_HIDE_FROM_ABI void wait_parity(bool __phase_parity) const
  {
    _CUDA_VSTD::__libcpp_thread_poll_with_backoff(
      _CUDA_VSTD::__barrier_poll_tester_parity<barrier>(this, __phase_parity));
  }

  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_wait()
  {
    wait(arrive());
  }

  _LIBCUDACXX_HIDE_FROM_ABI void arrive_and_drop()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!__isClusterShared(&__barrier)) { return __barrier.arrive_and_drop(); } else if (!__isShared(&__barrier)) {
          __trap();
        }

        asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(static_cast<_CUDA_VSTD::uint32_t>(
          __cvta_generic_to_shared(&__barrier)))
                     : "memory");),
      NV_PROVIDES_SM_80,
      (
        // Fallback to slowpath on device
        if (!__isShared(&__barrier)) {
          __barrier.arrive_and_drop();
          return;
        }

        asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(static_cast<_CUDA_VSTD::uint32_t>(
          __cvta_generic_to_shared(&__barrier)))
                     : "memory");),
      NV_ANY_TARGET,
      (
        // Fallback to slowpath on device
        __barrier.arrive_and_drop();))
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return (1 << 20) - 1;
  }

  template <class _Rep, class _Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_for(arrival_token&& __token, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);

    return __try_wait(_CUDA_VSTD::move(__token), __nanosec);
  }

  template <class _Clock, class _Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_until(arrival_token&& __token, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_for(_CUDA_VSTD::move(__token), (__time - _Clock::now()));
  }

  template <class _Rep, class _Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_parity_for(bool __phase_parity, const _CUDA_VSTD::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = _CUDA_VSTD::chrono::duration_cast<_CUDA_VSTD::chrono::nanoseconds>(__dur);

    return __try_wait_parity(__phase_parity, __nanosec);
  }

  template <class _Clock, class _Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool
  try_wait_parity_until(bool __phase_parity, const _CUDA_VSTD::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()));
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_BARRIER_BLOCK_SCOPE_H
