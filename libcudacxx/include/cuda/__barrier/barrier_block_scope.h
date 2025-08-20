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
#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/__memory/address_space.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/poll_tester.h>
#include <cuda/std/__new_>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !(&(((type*) 0)->member))
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !offsetof(type, member)
#endif // _CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Needed for pipeline.arrive_on
struct __block_scope_barrier_base
{};

template <>
class barrier<thread_scope_block, ::cuda::std::__empty_completion> : public __block_scope_barrier_base
{
  using __barrier_base = ::cuda::std::__barrier_base<::cuda::std::__empty_completion, thread_scope_block>;
  __barrier_base __barrier;

  _CCCL_DEVICE friend inline ::cuda::std::uint64_t* ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(
    barrier<thread_scope_block>& b);

  template <typename _Barrier>
  friend class ::cuda::std::__barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class ::cuda::std::__barrier_poll_tester_parity;

public:
  using arrival_token           = typename __barrier_base::arrival_token;
  _CCCL_HIDE_FROM_ABI barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  _CCCL_API inline barrier(::cuda::std::ptrdiff_t __expected,
                           ::cuda::std::__empty_completion __completion = ::cuda::std::__empty_completion())
  {
    static_assert(_LIBCUDACXX_OFFSET_IS_ZERO(barrier<thread_scope_block>, __barrier),
                  "fatal error: bad barrier layout");
    init(this, __expected, __completion);
  }

  _CCCL_API inline ~barrier()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
            ::__cvta_generic_to_shared(&__barrier)))
                       : "memory");
        } else if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          ::__trap();
        }),
      NV_PROVIDES_SM_80,
      (if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
        asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
          ::__cvta_generic_to_shared(&__barrier)))
                     : "memory");
      }))
  }

  _CCCL_API inline friend void init(barrier* __b,
                                    ::cuda::std::ptrdiff_t __expected,
                                    ::cuda::std::__empty_completion = ::cuda::std::__empty_completion())
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (::cuda::device::is_object_from(__b->__barrier, ::cuda::device::address_space::shared)) {
          asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(
                         static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__b->__barrier))),
                       "r"(static_cast<::cuda::std::uint32_t>(__expected))
                       : "memory");
        } else if (::cuda::device::is_object_from(__b->__barrier, ::cuda::device::address_space::cluster_shared)) {
          ::__trap();
        } else { new (&__b->__barrier) __barrier_base(__expected); }),
      NV_PROVIDES_SM_80,
      (
        if (::cuda::device::is_object_from(__b->__barrier, ::cuda::device::address_space::shared)) {
          asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(
                         static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__b->__barrier))),
                       "r"(static_cast<::cuda::std::uint32_t>(__expected))
                       : "memory");
        } else { new (&__b->__barrier) __barrier_base(__expected); }),
      NV_ANY_TARGET,
      (new (&__b->__barrier) __barrier_base(__expected);))
  }

  [[nodiscard]] _CCCL_API inline arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update >= 0, "Arrival count update must be non-negative.");
    arrival_token __token = {};
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return __barrier.arrive(__update);
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) { ::__trap(); }
        // Cannot use cuda::device::barrier_native_handle here, as it is
        // only defined for block-scope barriers. This barrier may be a
        // non-block scoped barrier.
        auto __bh = reinterpret_cast<::cuda::std::uint64_t*>(&__barrier);
        __token   = ::cuda::ptx::mbarrier_arrive(__bh, __update);),
      NV_PROVIDES_SM_80,
      (
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          return __barrier.arrive(__update);
        } auto __bh = reinterpret_cast<::cuda::std::uint64_t*>(&__barrier);
        // Need 2 instructions, can't finish barrier with arrive > 1
        if (__update > 1) { ::cuda::ptx::mbarrier_arrive_no_complete(__bh, __update - 1); } __token =
          ::cuda::ptx::mbarrier_arrive(__bh);),
      NV_PROVIDES_SM_70,
      (
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          return __barrier.arrive(__update);
        }

        unsigned int __mask    = ::__activemask();
        unsigned int __activeA = ::__match_any_sync(__mask, __update);
        unsigned int __activeB = ::__match_any_sync(__mask, reinterpret_cast<::cuda::std::uintptr_t>(&__barrier));
        unsigned int __active  = __activeA & __activeB;
        int __inc              = ::__popc(__active) * __update;

        int __leader = ::__ffs(__active) - 1;
        // All threads in mask synchronize here, establishing cummulativity to the __leader:
        ::__syncwarp(__mask);
        if (__leader == static_cast<int>(::cuda::ptx::get_sreg_laneid())) {
          __token = __barrier.arrive(__inc);
        } __token = ::__shfl_sync(__active, __token, __leader);),
      NV_IS_HOST,
      (__token = __barrier.arrive(__update);))
    return __token;
  }

private:
  _CCCL_API inline bool __test_wait_sm_80([[maybe_unused]] arrival_token __token) const
  {
    int32_t __ready = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_80,
      (asm volatile(
         "{\n\t"
         ".reg .pred p;\n\t"
         "mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
         "selp.b32 %0, 1, 0, p;\n\t"
         "}" : "=r"(__ready) : "r"(static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))),
         "l"(__token) : "memory");))
    return __ready;
  }

  // Document de drop > uint32_t for __nanosec on public for APIs
  _CCCL_API inline bool __try_wait([[maybe_unused]] arrival_token __token) const
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0;
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          ::__trap();
        } asm volatile("{\n\t"
                       ".reg .pred p;\n\t"
                       "mbarrier.try_wait.shared.b64 p, [%1], %2;\n\t"
                       "selp.b32 %0, 1, 0, p;\n\t"
                       "}" : "=r"(__ready) : "r"(
                         static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))),
                       "l"(__token) : "memory");
        return __ready;),
      NV_PROVIDES_SM_80,
      (if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
        return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));
      } return __test_wait_sm_80(__token);),
      NV_ANY_TARGET,
      (return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));))
  }

  // Document de drop > uint32_t for __nanosec on public for APIs
  _CCCL_API inline bool __try_wait(arrival_token __token, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait(::cuda::std::move(__token));
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0;
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return ::cuda::std::__cccl_thread_poll_with_backoff(
            ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)), __nanosec);
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) { ::__trap(); }

        ::cuda::std::chrono::high_resolution_clock::time_point const __start =
          ::cuda::std::chrono::high_resolution_clock::now();
        ::cuda::std::chrono::nanoseconds __elapsed;
        do {
          const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
          asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(__ready)
            : "r"(static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))),
              "l"(__token),
              "r"(__wait_nsec)
            : "memory");
          __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
        } while (!__ready && (__nanosec > __elapsed));
        return __ready;),
      NV_PROVIDES_SM_80,
      (
        bool __ready = 0;
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          return ::cuda::std::__cccl_thread_poll_with_backoff(
            ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)), __nanosec);
        }

        ::cuda::std::chrono::high_resolution_clock::time_point const __start =
          ::cuda::std::chrono::high_resolution_clock::now();
        do {
          __ready = __test_wait_sm_80(__token);
        } while (!__ready && __nanosec > (::cuda::std::chrono::high_resolution_clock::now() - __start));
        return __ready;),
      NV_ANY_TARGET,
      (return ::cuda::std::__cccl_thread_poll_with_backoff(
                ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)),
                ::cuda::std::chrono::nanoseconds(__nanosec));))
  }

  _CCCL_API inline bool __test_wait_parity_sm_80([[maybe_unused]] bool __phase_parity) const
  {
    uint16_t __ready = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_80,
      (asm volatile("{"
                    ".reg .pred %%p;"
                    "mbarrier.test_wait.parity.shared.b64 %%p, [%1], %2;"
                    "selp.u16 %0, 1, 0, %%p;"
                    "}" : "=h"(__ready) : "r"(static_cast<uint32_t>(::__cvta_generic_to_shared(&__barrier))),
                    "r"(static_cast<uint32_t>(__phase_parity)) : "memory");))
    return __ready;
  }

  _CCCL_API inline bool __try_wait_parity(bool __phase_parity) const
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          ::__trap();
        } int32_t __ready = 0;

        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
          "selp.b32 %0, 1, 0, p;\n\t"
          "}" : "=r"(__ready) : "r"(static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))),
          "r"(static_cast<::cuda::std::uint32_t>(__phase_parity)) :);

        return __ready;),
      NV_PROVIDES_SM_80,
      (if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
        return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);
      }

       return __test_wait_parity_sm_80(__phase_parity);),
      NV_ANY_TARGET,
      (return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);))
  }

  _CCCL_API inline bool __try_wait_parity(bool __phase_parity, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait_parity(__phase_parity);
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        int32_t __ready = 0;
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return ::cuda::std::__cccl_thread_poll_with_backoff(
            ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) { ::__trap(); }

        ::cuda::std::chrono::high_resolution_clock::time_point const __start =
          ::cuda::std::chrono::high_resolution_clock::now();
        ::cuda::std::chrono::nanoseconds __elapsed;
        do {
          const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
          asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2, %3;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(__ready)
            : "r"(static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))),
              "r"(static_cast<::cuda::std::uint32_t>(__phase_parity)),
              "r"(__wait_nsec)
            : "memory");
          __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
        } while (!__ready && (__nanosec > __elapsed));

        return __ready;),
      NV_PROVIDES_SM_80,
      (
        bool __ready = 0;
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          return ::cuda::std::__cccl_thread_poll_with_backoff(
            ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
        }

        ::cuda::std::chrono::high_resolution_clock::time_point const __start =
          ::cuda::std::chrono::high_resolution_clock::now();
        do {
          __ready = __test_wait_parity_sm_80(__phase_parity);
        } while (!__ready && __nanosec > (::cuda::std::chrono::high_resolution_clock::now() - __start));

        return __ready;),
      NV_ANY_TARGET,
      (return ::cuda::std::__cccl_thread_poll_with_backoff(
                ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);))
    _CCCL_UNREACHABLE();
  }

public:
  _CCCL_API inline void wait(arrival_token&& __phase) const
  {
    ::cuda::std::__cccl_thread_poll_with_backoff(
      ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__phase)));
  }

  _CCCL_API inline void wait_parity(bool __phase_parity) const
  {
    ::cuda::std::__cccl_thread_poll_with_backoff(
      ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity));
  }

  _CCCL_API inline void arrive_and_wait()
  {
    wait(arrive());
  }

  _CCCL_API inline void arrive_and_drop()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared)) {
          return __barrier.arrive_and_drop();
        } else if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) { ::__trap(); }

        asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(
          static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))) : "memory");),
      NV_PROVIDES_SM_80,
      (
        // Fallback to slowpath on device
        if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
          __barrier.arrive_and_drop();
          return;
        }

        asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(
          static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&__barrier))) : "memory");),
      NV_ANY_TARGET,
      (
        // Fallback to slowpath on device
        __barrier.arrive_and_drop();))
  }

  _CCCL_API static constexpr ptrdiff_t max() noexcept
  {
    return (1 << 20) - 1;
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_API inline bool
  try_wait_for(arrival_token&& __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    return __try_wait(::cuda::std::move(__token), __nanosec);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_API inline bool
  try_wait_until(arrival_token&& __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_for(::cuda::std::move(__token), (__time - _Clock::now()));
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_API inline bool
  try_wait_parity_for(bool __phase_parity, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    return __try_wait_parity(__phase_parity, __nanosec);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_API inline bool
  try_wait_parity_until(bool __phase_parity, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()));
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_BARRIER_BLOCK_SCOPE_H
