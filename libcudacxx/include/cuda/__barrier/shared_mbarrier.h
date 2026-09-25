//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_MBARRIER_H
#define _CUDA___BARRIER_SHARED_MBARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/instructions/mbarrier_complete_tx.h>
#  include <cuda/__ptx/instructions/mbarrier_expect_tx.h>
#  include <cuda/__ptx/instructions/mbarrier_init.h>
#  include <cuda/__ptx/instructions/mbarrier_inval.h>
#  include <cuda/__ptx/instructions/mbarrier_wait.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
struct __mbarrier_wait_status
{
  bool __complete;
  bool __report_predicate;
  ::cuda::std::uint8_t __report_value;
};

class __shared_mbarrier_impl
{
  ::cuda::std::uint64_t __barrier_;

public:
  using __arrival_token = ::cuda::std::uint64_t;

  _CCCL_HIDE_FROM_ABI __shared_mbarrier_impl() = default;

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t& __storage_ref() noexcept
  {
    return __barrier_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API const ::cuda::std::uint64_t& __storage_ref() const noexcept
  {
    return __barrier_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t* __native_handle() const noexcept
  {
    return const_cast<::cuda::std::uint64_t*>(&__barrier_);
  }

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 940
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __init_status_reporting(::cuda::std::uint32_t __count) const
  {
    ::cuda::ptx::mbarrier_init(::cuda::ptx::layout_v1, __native_handle(), __count);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __inval() const
  {
    ::cuda::ptx::mbarrier_inval(__native_handle());
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE __arrival_token __arrive(::cuda::std::ptrdiff_t __update) const
  {
    return ::cuda::ptx::mbarrier_arrive(__native_handle(), static_cast<::cuda::std::uint32_t>(__update));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __test_wait(__arrival_token __token) const
  {
    return ::cuda::ptx::mbarrier_test_wait(::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __token);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait(__arrival_token __token) const
  {
    return ::cuda::ptx::mbarrier_try_wait(::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __token);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait(__arrival_token __token, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    return ::cuda::ptx::mbarrier_try_wait(
      ::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __token, __suspend_time_hint);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __test_wait_status(__arrival_token __token) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_test_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_status(__arrival_token __token) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_status(__arrival_token __token, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token,
      __suspend_time_hint);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __test_wait_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_test_wait_parity(
      ::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __phase);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __phase);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_phase(::cuda::std::uint32_t __phase, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    return ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, __native_handle(), __phase, __suspend_time_hint);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __test_wait_phase_status(::cuda::std::uint32_t __phase) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_test_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_phase_status(::cuda::std::uint32_t __phase) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_phase_status(::cuda::std::uint32_t __phase, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase,
      __suspend_time_hint);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __test_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_test_wait_parity(
      ::cuda::ptx::mbarrier_phase_conditional,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __native_handle(),
      __phase);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_conditional,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __native_handle(),
      __phase);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __arrive_and_drop() const
  {
    (void) ::cuda::ptx::mbarrier_arrive_drop(
      ::cuda::ptx::sem_release, ::cuda::ptx::scope_cta, ::cuda::ptx::space_shared, __native_handle(), 1);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __expect_tx(::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    ::cuda::ptx::mbarrier_expect_tx(
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__transaction_count_update));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE __arrival_token
  __arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    if (__arrive_count_update == 1)
    {
      return ::cuda::ptx::mbarrier_arrive_expect_tx(
        ::cuda::ptx::sem_release,
        ::cuda::ptx::scope_cta,
        ::cuda::ptx::space_shared,
        __native_handle(),
        static_cast<::cuda::std::uint32_t>(__transaction_count_update));
    }

    __expect_tx(__transaction_count_update);
    return __arrive(__arrive_count_update);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __complete_tx(::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    ::cuda::ptx::mbarrier_complete_tx(
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__transaction_count_update));
  }
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 940
};

static_assert(sizeof(__shared_mbarrier_impl) == sizeof(::cuda::std::uint64_t),
              "shared mbarrier implementation must remain a single mbarrier word");
static_assert(alignof(__shared_mbarrier_impl) == alignof(::cuda::std::uint64_t),
              "shared mbarrier implementation must keep uint64_t alignment");
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_SHARED_MBARRIER_H
