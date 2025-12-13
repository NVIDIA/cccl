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
#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__memory/address_space.h>
#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/instructions/mbarrier_init.h>
#  include <cuda/__ptx/instructions/mbarrier_inval.h>
#  include <cuda/__ptx/instructions/mbarrier_wait.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__barrier/poll_tester.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/high_resolution_clock.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !(&(((type*) 0)->member))
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv
#  define _LIBCUDACXX_OFFSET_IS_ZERO(type, member) !offsetof(type, member)
#endif // _CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE
[[nodiscard]] _CCCL_DEVICE ::cuda::std::uint64_t* barrier_native_handle(barrier<thread_scope_block>& __b);
_CCCL_END_NAMESPACE_CUDA_DEVICE

_CCCL_BEGIN_NAMESPACE_CUDA

// Needed for pipeline.arrive_on
struct __block_scope_barrier_base
{};

template <>
class barrier<thread_scope_block, ::cuda::std::__empty_completion> : public __block_scope_barrier_base
{
  using __barrier_base = ::cuda::std::__barrier_base<::cuda::std::__empty_completion, thread_scope_block>;
  __barrier_base __barrier;

  _CCCL_DEVICE friend ::cuda::std::uint64_t* ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(
    barrier<thread_scope_block>& __b);

  [[nodiscard]] _CCCL_DEVICE ::cuda::std::uint64_t* __native_handle() const
  {
    return ::cuda::device::barrier_native_handle(const_cast<barrier&>(*this));
  }

  template <typename _Barrier>
  friend class ::cuda::std::__barrier_poll_tester_phase;
  template <typename _Barrier>
  friend class ::cuda::std::__barrier_poll_tester_parity;

public:
  using arrival_token           = typename __barrier_base::arrival_token;
  _CCCL_HIDE_FROM_ABI barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  _CCCL_API barrier(::cuda::std::ptrdiff_t __expected,
                    ::cuda::std::__empty_completion __completion = ::cuda::std::__empty_completion())
  {
    static_assert(_LIBCUDACXX_OFFSET_IS_ZERO(barrier<thread_scope_block>, __barrier),
                  "fatal error: bad barrier layout");
    init(this, __expected, __completion);
  }

  _CCCL_API ~barrier()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
                   ::cuda::ptx::mbarrier_inval(__native_handle());
                   return;
                 }))

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (_CCCL_ASSERT(!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared),
                    "barrier must not be in cluster shared memory");))
  }

  _CCCL_API inline friend void init(barrier* __b,
                                    ::cuda::std::ptrdiff_t __expected,
                                    ::cuda::std::__empty_completion = ::cuda::std::__empty_completion())
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (if (::cuda::device::is_object_from(__b->__barrier, ::cuda::device::address_space::shared)) {
                   ::cuda::ptx::mbarrier_init(__b->__native_handle(), static_cast<::cuda::std::uint32_t>(__expected));
                   return;
                 }))

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (_CCCL_ASSERT(!::cuda::device::is_object_from(__b->__barrier, ::cuda::device::address_space::cluster_shared),
                    "barrier must not be in cluster shared memory");))

    new (&__b->__barrier) __barrier_base(__expected);
  }

private:
#if _CCCL_CUDA_COMPILATION()
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token __arrive_sm90(::cuda::std::ptrdiff_t __update)
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return __barrier.arrive(__update);
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must be in shared memory, not cluster shared memory");
    return ::cuda::ptx::mbarrier_arrive(__native_handle(), __update);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token __arrive_sm80(::cuda::std::ptrdiff_t __update)
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return __barrier.arrive(__update);
    }
    // Need 2 instructions, can't finish barrier with arrive > 1
    if (__update > 1)
    {
      ::cuda::ptx::mbarrier_arrive_no_complete(__native_handle(), __update - 1);
    }
    return ::cuda::ptx::mbarrier_arrive(__native_handle());
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token __arrive_sm70(::cuda::std::ptrdiff_t __update)
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return __barrier.arrive(__update);
    }

    unsigned int __mask    = ::__activemask();
    unsigned int __activeA = ::__match_any_sync(__mask, __update);
    unsigned int __activeB = ::__match_any_sync(__mask, reinterpret_cast<::cuda::std::uintptr_t>(&__barrier));
    unsigned int __active  = __activeA & __activeB;
    int __inc              = ::cuda::std::popcount(__active) * __update;

    int __leader = ::__ffs(__active) - 1;
    // All threads in mask synchronize here, establishing cummulativity to the __leader:
    ::__syncwarp(__mask);
    arrival_token __token = {};
    if (__leader == static_cast<int>(::cuda::ptx::get_sreg_laneid()))
    {
      __token = __barrier.arrive(__inc);
    }
    return ::__shfl_sync(__active, __token, __leader);
  }
#endif // _CCCL_CUDA_COMPILATION()

public:
  /*discard*/ _CCCL_API arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(__update >= 0, "Arrival count update must be non-negative.");
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return __arrive_sm90(__update);),
      NV_PROVIDES_SM_80,
      (return __arrive_sm80(__update);),
      NV_PROVIDES_SM_70,
      (return __arrive_sm70(__update);),
      NV_IS_HOST,
      (return __barrier.arrive(__update);))
  }

private:
#if _CCCL_CUDA_COMPILATION()
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_sm90(arrival_token __token) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must be in shared memory, not cluster shared memory");
    return ::cuda::ptx::mbarrier_try_wait(__native_handle(), __token);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_sm80(arrival_token __token) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));
    }
    return ::cuda::ptx::mbarrier_test_wait(__native_handle(), __token);
  }
#endif // _CCCL_CUDA_COMPILATION()

  [[nodiscard]] _CCCL_API bool __try_wait(arrival_token __token) const
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return __try_wait_sm90(__token);),
      NV_PROVIDES_SM_80,
      (return __try_wait_sm80(__token);),
      NV_ANY_TARGET,
      (return ::cuda::std::__call_try_wait(__barrier, ::cuda::std::move(__token));))
  }

#if _CCCL_CUDA_COMPILATION()
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_sm90(arrival_token __token, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return ::cuda::std::__cccl_thread_poll_with_backoff(
        ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)), __nanosec);
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must not be in cluster shared memory");

    bool __ready = 0;
    ::cuda::std::chrono::high_resolution_clock::time_point const __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    ::cuda::std::chrono::nanoseconds __elapsed;
    do
    {
      const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
      ::cuda::ptx::mbarrier_try_wait(__native_handle(), __token, __wait_nsec);
      __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    } while (!__ready && (__nanosec > __elapsed));
    return __ready;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_sm80(arrival_token __token, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return ::cuda::std::__cccl_thread_poll_with_backoff(
        ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)), __nanosec);
    }

    bool __ready = false;
    ::cuda::std::chrono::high_resolution_clock::time_point const __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    do
    {
      __ready = ::cuda::ptx::mbarrier_test_wait(__native_handle(), __token);
    } while (!__ready && __nanosec > (::cuda::std::chrono::high_resolution_clock::now() - __start));
    return __ready;
  }
#endif // _CCCL_CUDA_COMPILATION()

  // Document de drop > uint32_t for __nanosec on public for APIs
  [[nodiscard]] _CCCL_API bool __try_wait(arrival_token __token, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait(::cuda::std::move(__token));
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return __try_wait_sm90(__token, __nanosec);),
      NV_PROVIDES_SM_80,
      (return __try_wait_sm80(__token, __nanosec);),
      NV_ANY_TARGET,
      (return ::cuda::std::__cccl_thread_poll_with_backoff(
                ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__token)),
                ::cuda::std::chrono::nanoseconds(__nanosec));))
  }

#if _CCCL_CUDA_COMPILATION()
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_parity_sm90(bool __phase_parity) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must be in shared memory, not cluster shared memory");

    return ::cuda::ptx::mbarrier_try_wait_parity(__native_handle(), __phase_parity);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_parity_sm80(bool __phase_parity) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);
    }
    return ::cuda::ptx::mbarrier_test_wait_parity(__native_handle(), __phase_parity);
  }
#endif // _CCCL_CUDA_COMPILATION()

  [[nodiscard]] _CCCL_API bool __try_wait_parity(bool __phase_parity) const
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return __try_wait_parity_sm90(__phase_parity);),
      NV_PROVIDES_SM_80,
      (return __try_wait_parity_sm80(__phase_parity);),
      NV_ANY_TARGET,
      (return ::cuda::std::__call_try_wait_parity(__barrier, __phase_parity);))
  }

#if _CCCL_CUDA_COMPILATION()
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_parity_sm90(bool __phase_parity, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return ::cuda::std::__cccl_thread_poll_with_backoff(
        ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must be in shared memory, not cluster shared memory");

    int32_t __ready = 0;
    ::cuda::std::chrono::high_resolution_clock::time_point const __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    ::cuda::std::chrono::nanoseconds __elapsed;
    do
    {
      const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
      ::cuda::ptx::mbarrier_try_wait_parity(__native_handle(), __phase_parity, __wait_nsec);
      __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    } while (!__ready && (__nanosec > __elapsed));

    return __ready;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_parity_sm80(bool __phase_parity, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      return ::cuda::std::__cccl_thread_poll_with_backoff(
        ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);
    }

    bool __ready = 0;
    ::cuda::std::chrono::high_resolution_clock::time_point const __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    do
    {
      __ready = ::cuda::ptx::mbarrier_test_wait_parity(__native_handle(), __phase_parity);
    } while (!__ready && __nanosec > (::cuda::std::chrono::high_resolution_clock::now() - __start));

    return __ready;
  }
#endif // _CCCL_CUDA_COMPILATION()

  [[nodiscard]] _CCCL_API bool __try_wait_parity(bool __phase_parity, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait_parity(__phase_parity);
    }

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return __try_wait_parity_sm90(__phase_parity, __nanosec);),
      NV_PROVIDES_SM_80,
      (return __try_wait_parity_sm80(__phase_parity, __nanosec);),
      NV_ANY_TARGET,
      (return ::cuda::std::__cccl_thread_poll_with_backoff(
                ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity), __nanosec);))
  }

public:
  _CCCL_API void wait(arrival_token&& __phase) const
  {
    // no need to back off on a barrier in SMEM on SM90+, SYNCS unit is taking care of this
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
                   while (!::cuda::ptx::mbarrier_try_wait(
                     reinterpret_cast<uint64_t*>(const_cast<__barrier_base*>(&__barrier)), __phase))
                     ;
                   return;
                 }))
    // fallback implementation
    ::cuda::std::__cccl_thread_poll_with_backoff(
      ::cuda::std::__barrier_poll_tester_phase<barrier>(this, ::cuda::std::move(__phase)));
  }

  _CCCL_API void wait_parity(bool __phase_parity) const
  {
    // no need to back off on a barrier in SMEM on SM90+, SYNCS unit is taking care of this
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (if (::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared)) {
                   while (!::cuda::ptx::mbarrier_try_wait_parity(
                     reinterpret_cast<uint64_t*>(const_cast<__barrier_base*>(&__barrier)), __phase_parity))
                     ;
                   return;
                 }))
    // fallback implementation
    ::cuda::std::__cccl_thread_poll_with_backoff(
      ::cuda::std::__barrier_poll_tester_parity<barrier>(this, __phase_parity));
  }

  _CCCL_API void arrive_and_wait()
  {
    wait(arrive());
  }

private:
#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __arrive_and_drop_sm90()
  {
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::cluster_shared))
    {
      return __barrier.arrive_and_drop();
    }
    _CCCL_ASSERT(::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared),
                 "barrier must be in shared memory, not cluster shared memory");

    // TODO(bgruber): expose mbarrier.arrive_drop.shared in cuda::ptx
    asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
      ::__cvta_generic_to_shared(&__barrier)))
                 : "memory");
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __arrive_and_drop_sm80()
  {
    // Fallback to slowpath on device
    if (!::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared))
    {
      __barrier.arrive_and_drop();
      return;
    }

    // TODO(bgruber): expose mbarrier.arrive_drop.shared in cuda::ptx
    asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
      ::__cvta_generic_to_shared(&__barrier)))
                 : "memory");
  }
#endif // _CCCL_CUDA_COMPILATION()

public:
  _CCCL_API void arrive_and_drop()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (__arrive_and_drop_sm90();),
      NV_PROVIDES_SM_80,
      (__arrive_and_drop_sm80();),
      // Fallback to slowpath on device
      NV_ANY_TARGET,
      (__barrier.arrive_and_drop();))
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::ptrdiff_t max() noexcept
  {
    return (1 << 20) - 1;
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_API bool
  try_wait_for(arrival_token&& __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    return __try_wait(::cuda::std::move(__token), __nanosec);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_API bool
  try_wait_until(arrival_token&& __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_for(::cuda::std::move(__token), (__time - _Clock::now()));
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_API bool
  try_wait_parity_for(bool __phase_parity, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur)
  {
    auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    return __try_wait_parity(__phase_parity, __nanosec);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_API bool
  try_wait_parity_until(bool __phase_parity, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time)
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()));
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE inline ::cuda::std::uint64_t* barrier_native_handle(barrier<thread_scope_block>& __b)
{
  return reinterpret_cast<::cuda::std::uint64_t*>(&__b.__barrier);
}
_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_BARRIER_BLOCK_SCOPE_H
