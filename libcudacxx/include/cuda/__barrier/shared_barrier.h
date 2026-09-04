//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_BARRIER_H
#define _CUDA___BARRIER_SHARED_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/barrier.h>
#include <cuda/__utility/status_policy.h>
#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__memory/address_space.h>
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/instructions/mbarrier_expect_tx.h>
#  include <cuda/__ptx/instructions/mbarrier_init.h>
#  include <cuda/__ptx/instructions/mbarrier_inval.h>
#  include <cuda/__ptx/instructions/mbarrier_wait.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/high_resolution_clock.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)
#  include <cuda_runtime_api.h>
#  define _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS 1
#else // ^^^ _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3) ^^^
#  define _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS 0
#endif // ^^^ !_CCCL_CUDA_COMPILATION() || _CCCL_COMPILER(NVRTC) || _CCCL_CUDACC_BELOW(13, 3) ^^^

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

enum class shared_barrier_kind
{
  completion_only,
  status_reporting,
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b);
_CCCL_END_NAMESPACE_CUDA_DEVICE

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
struct __mbarrier_wait_status
{
  bool __complete;
  bool __report_predicate;
  ::cuda::std::uint8_t __report_value;
};
} // namespace __detail

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
namespace __detail
{
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();

_CCCL_DEVICE_API inline void __mbarrier_init_layout_v1(::cuda::std::uint64_t* __addr, ::cuda::std::uint32_t __count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("mbarrier.init.layout::v1.shared::cta.b64 [%0], %1;"
      :
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline bool __mbarrier_check_layout_v1(::cuda::std::uint64_t* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __is_v1;
  asm("{\n\t"
      ".reg .pred P_OUT;\n\t"
      "mbarrier.check_layout.layout::v1.shared::cta.b64 P_OUT, [%1];\n\t"
      "selp.b32 %0, 1, 0, P_OUT;\n\t"
      "}"
      : "=r"(__is_v1)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr))
      : "memory");
  return static_cast<bool>(__is_v1);
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return false;
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline __mbarrier_wait_status
__mbarrier_test_wait_primary_status(::cuda::std::uint64_t* __addr, ::cuda::std::uint64_t __state)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  ::cuda::std::uint32_t __report_predicate;
  ::cuda::std::uint32_t __report_value;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      ".reg .pred P_REPORT;\n\t"
      ".reg .b8 R_REPORT;\n\t"
      "mbarrier.test_wait.phase_type::primary.shared::cta.b64 P_COMPLETE|P_REPORT, R_REPORT, [%3], %4;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "selp.b32 %1, 1, 0, P_REPORT;\n\t"
      "cvt.u32.u8 %2, R_REPORT;\n\t"
      "}"
      : "=r"(__complete), "=r"(__report_predicate), "=r"(__report_value)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "l"(__state)
      : "memory");
  return {static_cast<bool>(__complete),
          static_cast<bool>(__report_predicate),
          static_cast<::cuda::std::uint8_t>(__report_value)};
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return {};
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline __mbarrier_wait_status __mbarrier_try_wait_primary_status(
  ::cuda::std::uint64_t* __addr, ::cuda::std::uint64_t __state, ::cuda::std::uint32_t __suspend_time_hint)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  ::cuda::std::uint32_t __report_predicate;
  ::cuda::std::uint32_t __report_value;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      ".reg .pred P_REPORT;\n\t"
      ".reg .b8 R_REPORT;\n\t"
      "mbarrier.try_wait.phase_type::primary.shared::cta.b64 P_COMPLETE|P_REPORT, R_REPORT, [%3], %4, %5;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "selp.b32 %1, 1, 0, P_REPORT;\n\t"
      "cvt.u32.u8 %2, R_REPORT;\n\t"
      "}"
      : "=r"(__complete), "=r"(__report_predicate), "=r"(__report_value)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "l"(__state), "r"(__suspend_time_hint)
      : "memory");
  return {static_cast<bool>(__complete),
          static_cast<bool>(__report_predicate),
          static_cast<::cuda::std::uint8_t>(__report_value)};
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return {};
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline __mbarrier_wait_status
__mbarrier_test_wait_parity_primary_status(::cuda::std::uint64_t* __addr, ::cuda::std::uint32_t __phase_parity)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  ::cuda::std::uint32_t __report_predicate;
  ::cuda::std::uint32_t __report_value;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      ".reg .pred P_REPORT;\n\t"
      ".reg .b8 R_REPORT;\n\t"
      "mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 P_COMPLETE|P_REPORT, R_REPORT, [%3], %4;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "selp.b32 %1, 1, 0, P_REPORT;\n\t"
      "cvt.u32.u8 %2, R_REPORT;\n\t"
      "}"
      : "=r"(__complete), "=r"(__report_predicate), "=r"(__report_value)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "r"(__phase_parity)
      : "memory");
  return {static_cast<bool>(__complete),
          static_cast<bool>(__report_predicate),
          static_cast<::cuda::std::uint8_t>(__report_value)};
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return {};
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline __mbarrier_wait_status __mbarrier_try_wait_parity_primary_status(
  ::cuda::std::uint64_t* __addr, ::cuda::std::uint32_t __phase_parity, ::cuda::std::uint32_t __suspend_time_hint)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  ::cuda::std::uint32_t __report_predicate;
  ::cuda::std::uint32_t __report_value;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      ".reg .pred P_REPORT;\n\t"
      ".reg .b8 R_REPORT;\n\t"
      "mbarrier.try_wait.parity.phase_type::primary.shared::cta.b64 P_COMPLETE|P_REPORT, R_REPORT, [%3], %4, %5;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "selp.b32 %1, 1, 0, P_REPORT;\n\t"
      "cvt.u32.u8 %2, R_REPORT;\n\t"
      "}"
      : "=r"(__complete), "=r"(__report_predicate), "=r"(__report_value)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "r"(__phase_parity), "r"(__suspend_time_hint)
      : "memory");
  return {static_cast<bool>(__complete),
          static_cast<bool>(__report_predicate),
          static_cast<::cuda::std::uint8_t>(__report_value)};
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return {};
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline bool
__mbarrier_test_wait_parity_conditional(::cuda::std::uint64_t* __addr, ::cuda::std::uint32_t __phase_parity)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      "mbarrier.test_wait.parity.phase_type::conditional.shared::cta.b64 P_COMPLETE, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "}"
      : "=r"(__complete)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "r"(__phase_parity)
      : "memory");
  return static_cast<bool>(__complete);
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return false;
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}

[[nodiscard]] _CCCL_DEVICE_API inline bool __mbarrier_try_wait_parity_conditional(
  ::cuda::std::uint64_t* __addr, ::cuda::std::uint32_t __phase_parity, ::cuda::std::uint32_t __suspend_time_hint)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __complete;
  asm("{\n\t"
      ".reg .pred P_COMPLETE;\n\t"
      "mbarrier.try_wait.parity.phase_type::conditional.shared::cta.b64 P_COMPLETE, [%1], %2, %3;\n\t"
      "selp.b32 %0, 1, 0, P_COMPLETE;\n\t"
      "}"
      : "=r"(__complete)
      : "r"(::cuda::ptx::__as_ptr_smem(__addr)), "r"(__phase_parity), "r"(__suspend_time_hint)
      : "memory");
  return static_cast<bool>(__complete);
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900 ^^^
  ::cuda::__detail::__cuda_ptx_mbarrier_layout_v1_is_not_supported_before_SM_90__();
  return false;
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) && __CUDA_ARCH__ < 900 ^^^
}
} // namespace __detail
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930

namespace __detail
{
struct __issue_and_wait_access;
} // namespace __detail

class shared_barrier
{
  ::cuda::std::uint64_t __barrier_;

  _CCCL_DEVICE_API friend ::cuda::std::uint64_t* ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(
    ::cuda::shared_barrier& __b);

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* __native_handle() const
  {
    return const_cast<::cuda::std::uint64_t*>(&__barrier_);
  }

public:
  class operation_status
  {
    bool __complete_                     = false;
    bool __report_predicate_             = false;
    ::cuda::std::uint8_t __report_value_ = 0;
    mutable bool __report_inspected_     = false;

    friend struct ::cuda::__detail::__issue_and_wait_access;

    _CCCL_HOST_DEVICE_API constexpr operation_status(
      bool __complete, bool __report_predicate, ::cuda::std::uint8_t __report_value) noexcept
        : __complete_(__complete)
        , __report_predicate_(__report_predicate)
        , __report_value_(__report_value)
    {}

    friend class shared_barrier;

    _CCCL_HOST_DEVICE_API void __assert_report_inspected() const noexcept
    {
      if (__report_predicate_ && !__report_inspected_)
      {
        NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
        _CCCL_UNREACHABLE();
      }
    }

    _CCCL_HOST_DEVICE_API void __mark_report_inspected() const noexcept
    {
      __report_inspected_ = true;
    }

  public:
    _CCCL_HOST_DEVICE_API constexpr operation_status() noexcept {}

    operation_status(const operation_status&)            = delete;
    operation_status& operator=(const operation_status&) = delete;

    _CCCL_HOST_DEVICE_API operation_status(operation_status&& __other) noexcept
        : __complete_(__other.__complete_)
        , __report_predicate_(__other.__report_predicate_)
        , __report_value_(__other.__report_value_)
        , __report_inspected_(__other.__report_inspected_)
    {
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
    }

    _CCCL_HOST_DEVICE_API operation_status& operator=(operation_status&& __other) noexcept
    {
      __assert_report_inspected();
      __complete_                 = __other.__complete_;
      __report_predicate_         = __other.__report_predicate_;
      __report_value_             = __other.__report_value_;
      __report_inspected_         = __other.__report_inspected_;
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
      return *this;
    }

    _CCCL_HOST_DEVICE_API ~operation_status() noexcept
    {
      __assert_report_inspected();
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool complete() const noexcept
    {
      return __complete_;
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API bool has_report() const noexcept
    {
      if (__report_predicate_)
      {
        __mark_report_inspected();
      }
      return __report_predicate_;
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator bool() const noexcept
    {
      const bool __has_report = has_report();
      return complete() && !__has_report;
    }

  private:
#if _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS
    _CCCL_DEVICE_API static void __assert_fabric_status(::cudaError_t __status) noexcept
    {
      _CCCL_ASSERT(__status == ::cudaSuccess, "failed to decode shared_barrier status");
      if (__status != ::cudaSuccess)
      {
        ::cuda::std::terminate();
      }
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API static bool __encodes_fabric_errors(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::tma_validity_check:
          return false;
        case status_source::fabric_push_reduction:
          return true;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API static ::cudaFabricOpStatusSource
    __cuda_status_source(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::fabric_push_reduction:
          return ::cudaFabricOpStatusSourceMbarrierV1;
        case status_source::tma_validity_check:
          _CCCL_ASSERT(false, "shared_barrier status source does not encode fabric errors");
          return ::cudaFabricOpStatusSourceMbarrierV1;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned int __error_count(status_source __source) const noexcept
    {
      if (!__encodes_fabric_errors(__source))
      {
        return 0;
      }
      unsigned int __count = 0;
      auto __report_value  = __report_value_;
      __assert_fabric_status(::cudaFabricOpErrorStatusCount(&__report_value, __cuda_status_source(__source), &__count));
      return __count;
    }

    [[nodiscard]] _CCCL_DEVICE_API ::cudaFabricOpStatusInfo
    __error_status(status_source __source, unsigned int __status_index) const noexcept
    {
      _CCCL_ASSERT(__encodes_fabric_errors(__source), "shared_barrier status source does not encode fabric errors");
      ::cudaFabricOpStatusInfo __status_info{};
      auto __report_value = __report_value_;
      __assert_fabric_status(
        ::cudaFabricOpErrorStatusGet(&__report_value, __cuda_status_source(__source), __status_index, &__status_info));
      return __status_info;
    }
#else // ^^^ _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS ^^^
    [[nodiscard]] _CCCL_HOST_DEVICE_API static bool __encodes_fabric_errors(status_source) noexcept
    {
      return false;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned int __error_count(status_source) const noexcept
    {
      return 0;
    }
#endif // ^^^ !_LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS ^^^

  public:
    [[nodiscard]] _CCCL_DEVICE_API unsigned int get_error_count(status_source __source) const noexcept
    {
      __mark_report_inspected();
      return __error_count(__source);
    }

    template <class _Fn>
    _CCCL_DEVICE_API void for_each_error(status_source __source, _Fn __fn) const noexcept
    {
      __mark_report_inspected();
      const unsigned int __count = __error_count(__source);
      for (unsigned int __index = 0; __index != __count; ++__index)
      {
#if _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS
        __fn(__error_status(__source, __index));
#else // ^^^ _LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS ^^^
        (void) __index;
        (void) __fn;
#endif // ^^^ !_LIBCUDACXX_HAS_CUDA_FABRIC_OP_STATUS ^^^
      }
    }

  private:
    [[nodiscard]] _CCCL_DEVICE_API bool __should_reissue_reported_operation(status_source __source) const noexcept
    {
      if (!__report_predicate_)
      {
        return false;
      }
      if (!__encodes_fabric_errors(__source))
      {
        __mark_report_inspected();
        return true;
      }
      if (__error_count(__source) != 0)
      {
        return false;
      }
      __mark_report_inspected();
      return true;
    }
  };

  using arrival_token                  = ::cuda::std::uint64_t;
  _CCCL_HIDE_FROM_ABI shared_barrier() = default;

  shared_barrier(const shared_barrier&)            = delete;
  shared_barrier& operator=(const shared_barrier&) = delete;

  _CCCL_HOST_DEVICE_API explicit shared_barrier(::cuda::std::ptrdiff_t __expected,
                                                shared_barrier_kind __kind = shared_barrier_kind::completion_only)
  {
    init(this, __kind, __expected);
  }

  _CCCL_HOST_DEVICE_API explicit shared_barrier(shared_barrier_kind __kind, ::cuda::std::ptrdiff_t __expected)
  {
    init(this, __kind, __expected);
  }

private:
  [[noreturn]] _CCCL_HOST_DEVICE_API static void __unsupported_storage() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier requires local shared memory and shared-memory mbarrier support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[noreturn]] _CCCL_HOST_DEVICE_API static void __unsupported_status_reporting() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier status_reporting kind requires mbarrier layout v1 support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[noreturn]] _CCCL_HOST_DEVICE_API static void __discarded_report() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier no-status wait discarded a status report");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t
  __max_for_kind(shared_barrier_kind __kind) noexcept
  {
    return __kind == shared_barrier_kind::status_reporting ? ((1 << 9) - 1) : ((1 << 20) - 1);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t __max_transaction_count_update() noexcept
  {
    return (1 << 20) - 1;
  }

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __assert_supported_storage() const
  {
    if (!::cuda::device::is_object_from(__barrier_, ::cuda::device::address_space::shared))
    {
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (_CCCL_ASSERT(!::cuda::device::is_object_from(__barrier_, ::cuda::device::address_space::cluster_shared),
                      "shared_barrier must not be in another block's cluster shared memory");))
      __unsupported_storage();
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static operation_status
  __make_operation_status(::cuda::__detail::__mbarrier_wait_status __result) noexcept
  {
    return operation_status(__result.__complete, __result.__report_predicate, __result.__report_value);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static ::cuda::std::uint32_t __phase_parity_value(bool __phase_parity) noexcept
  {
    return __phase_parity ? 1 : 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __make_completion_only_operation_status(bool __complete) const noexcept
  {
    return operation_status(__complete, false, 0);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __test_wait_completion_only(arrival_token __token) const
  {
    return __make_completion_only_operation_status(::cuda::ptx::mbarrier_test_wait(__native_handle(), __token));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __test_wait_parity_completion_only(bool __phase_parity) const
  {
    return __make_completion_only_operation_status(
      ::cuda::ptx::mbarrier_test_wait_parity(__native_handle(), __phase_parity_value(__phase_parity)));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token __arrive_mbarrier_sm80(::cuda::std::ptrdiff_t __update)
  {
    if (__update > 1)
    {
      ::cuda::ptx::mbarrier_arrive_no_complete(__native_handle(), static_cast<::cuda::std::uint32_t>(__update - 1));
    }
    return ::cuda::ptx::mbarrier_arrive(__native_handle());
  }

#  if __cccl_ptx_isa >= 930
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token __arrive_mbarrier_sm90(::cuda::std::ptrdiff_t __update)
  {
    return ::cuda::ptx::mbarrier_arrive(
      ::cuda::ptx::sem_release,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__update));
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __expect_tx_mbarrier(::cuda::std::ptrdiff_t __transaction_count_update)
  {
    ::cuda::ptx::mbarrier_expect_tx(
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__transaction_count_update));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE arrival_token
  __arrive_tx_mbarrier(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update)
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

    __expect_tx_mbarrier(__transaction_count_update);
    return __arrive_mbarrier_sm90(__arrive_count_update);
  }
#  endif // __cccl_ptx_isa >= 930

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __arrive_and_drop_mbarrier()
  {
    asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
      ::__cvta_generic_to_shared(__native_handle())))
                 : "memory");
  }
#endif // _CCCL_CUDA_COMPILATION()

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::ptrdiff_t __max_for_current_kind() const
  {
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (return __max_for_kind(is_kind(shared_barrier_kind::status_reporting) ? shared_barrier_kind::status_reporting
                                                                            : shared_barrier_kind::completion_only);))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__assert_supported_storage(); return __max_for_kind(shared_barrier_kind::completion_only);))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool __completed_without_report(const operation_status& __status) const
  {
    if (__status.has_report())
    {
      __discarded_report();
    }
    return __status.complete();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool __completed_ignoring_report(const operation_status& __status) const
  {
    (void) __status.has_report();
    return __status.complete();
  }

public:
  _CCCL_HOST_DEVICE_API ~shared_barrier()
  {
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__assert_supported_storage(); ::cuda::ptx::mbarrier_inval(__native_handle()); return;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API inline friend void
  init(shared_barrier* __b, shared_barrier_kind __kind, ::cuda::std::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(1 <= __expected, "Expected arrival count must be at least one.");
    _CCCL_ASSERT(__expected <= shared_barrier::max(__kind),
                 "Expected arrival count cannot exceed shared_barrier::max(kind).");

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (
                   __b->__assert_supported_storage(); if (__kind == shared_barrier_kind::status_reporting) {
                     ::cuda::__detail::__mbarrier_init_layout_v1(
                       __b->__native_handle(), static_cast<::cuda::std::uint32_t>(__expected));
                   } else {
                     ::cuda::ptx::mbarrier_init(__b->__native_handle(), static_cast<::cuda::std::uint32_t>(__expected));
                   } return;))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__b->__assert_supported_storage(); if (__kind == shared_barrier_kind::status_reporting) {
                   __unsupported_status_reporting();
                 } ::cuda::ptx::mbarrier_init(__b->__native_handle(), static_cast<::cuda::std::uint32_t>(__expected));
                  return;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API inline friend void init(shared_barrier* __b, ::cuda::std::ptrdiff_t __expected)
  {
    init(__b, shared_barrier_kind::completion_only, __expected);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool is_kind(shared_barrier_kind __kind) const
  {
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (__assert_supported_storage();
                  const bool __is_v1 = ::cuda::__detail::__mbarrier_check_layout_v1(__native_handle());
                  return __kind == shared_barrier_kind::status_reporting ? __is_v1 : !__is_v1;))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__assert_supported_storage(); return __kind == shared_barrier_kind::completion_only;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(1 <= __update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__update <= __max_for_current_kind(),
                 "Arrival count update cannot exceed shared_barrier::max(active kind).");

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__assert_supported_storage(); return __arrive_mbarrier_sm90(__update);))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80, (__assert_supported_storage(); return __arrive_mbarrier_sm80(__update);))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API void expect_tx(::cuda::std::ptrdiff_t __transaction_count_update)
  {
    _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
    _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
                 "Transaction count update cannot exceed the mbarrier transaction count limit.");

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (__assert_supported_storage(); __expect_tx_mbarrier(__transaction_count_update); return;))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token
  arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update)
  {
    _CCCL_ASSERT(1 <= __arrive_count_update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__arrive_count_update <= __max_for_current_kind(),
                 "Arrival count update cannot exceed shared_barrier::max(active kind).");
    _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
    _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
                 "Transaction count update cannot exceed the mbarrier transaction count limit.");

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__assert_supported_storage(); return __arrive_tx_mbarrier(__arrive_count_update, __transaction_count_update);))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(arrival_token __token, return_status_t) const
  {
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__assert_supported_storage();
       const auto __result = ::cuda::__detail::__mbarrier_test_wait_primary_status(__native_handle(), __token);
       return __make_operation_status(__result);))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80, (__assert_supported_storage(); return __test_wait_completion_only(__token);))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(arrival_token __token) const
  {
    const operation_status __status = try_wait(__token, return_status);
    return __completed_without_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(arrival_token __token, ignore_status_t) const
  {
    const operation_status __status = try_wait(__token, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(arrival_token __token, return_status_t) const
  {
    operation_status __result;
    do
    {
      __result = try_wait(__token, return_status);
    } while (!__result.complete());
    return __result;
  }

  _CCCL_HOST_DEVICE_API void wait(arrival_token __token) const
  {
    while (!try_wait(__token))
    {
    }
  }

  _CCCL_HOST_DEVICE_API void wait(arrival_token __token, ignore_status_t) const
  {
    while (!try_wait(__token, ignore_status))
    {
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status arrive_and_wait(return_status_t)
  {
    return wait(arrive(), return_status);
  }

  _CCCL_HOST_DEVICE_API void arrive_and_wait()
  {
    wait(arrive());
  }

  _CCCL_HOST_DEVICE_API void arrive_and_wait(ignore_status_t)
  {
    wait(arrive(), ignore_status);
  }

  _CCCL_HOST_DEVICE_API void arrive_and_drop()
  {
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80, (__assert_supported_storage(); __arrive_and_drop_mbarrier(); return;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_parity(bool __phase_parity, return_status_t) const
  {
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (__assert_supported_storage();
                  const auto __result = ::cuda::__detail::__mbarrier_test_wait_parity_primary_status(
                    __native_handle(), __phase_parity_value(__phase_parity));
                  return __make_operation_status(__result);))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__assert_supported_storage(); return __test_wait_parity_completion_only(__phase_parity);))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_parity(bool __phase_parity) const
  {
    const operation_status __status = try_wait_parity(__phase_parity, return_status);
    return __completed_without_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_parity(bool __phase_parity, ignore_status_t) const
  {
    const operation_status __status = try_wait_parity(__phase_parity, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait_parity(bool __phase_parity, return_status_t) const
  {
    operation_status __result;
    do
    {
      __result = try_wait_parity(__phase_parity, return_status);
    } while (!__result.complete());
    return __result;
  }

  _CCCL_HOST_DEVICE_API void wait_parity(bool __phase_parity) const
  {
    while (!try_wait_parity(__phase_parity))
    {
    }
  }

  _CCCL_HOST_DEVICE_API void wait_parity(bool __phase_parity, ignore_status_t) const
  {
    while (!try_wait_parity(__phase_parity, ignore_status))
    {
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_conditional_phase_parity(bool __phase_parity) const
  {
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (__assert_supported_storage(); return ::cuda::__detail::__mbarrier_test_wait_parity_conditional(
                    __native_handle(), __phase_parity_value(__phase_parity));))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (__assert_supported_storage(); return static_cast<bool>(__test_wait_parity_completion_only(__phase_parity));))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API void wait_conditional_phase_parity(bool __phase_parity) const
  {
    while (!try_wait_conditional_phase_parity(__phase_parity))
    {
    }
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return try_wait(__token, return_status);
    }

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__assert_supported_storage(); operation_status __result;
       const ::cuda::std::chrono::high_resolution_clock::time_point __start =
         ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         const auto __ptx_result =
           ::cuda::__detail::__mbarrier_try_wait_primary_status(__native_handle(), __token, __wait_nsec);
         __result  = __make_operation_status(__ptx_result);
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (__assert_supported_storage(); operation_status __result;
       const ::cuda::std::chrono::high_resolution_clock::time_point __start =
         ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         __result  = __test_wait_completion_only(__token);
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur) const
  {
    const operation_status __status = try_wait_for(__token, __dur, return_status);
    return __completed_without_report(__status);
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const operation_status __status = try_wait_for(__token, __dur, return_status);
    return __completed_ignoring_report(__status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, return_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), return_status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_until(arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time) const
  {
    return try_wait_for(__token, (__time - _Clock::now()));
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, ignore_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), ignore_status);
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_parity_for(
    bool __phase_parity, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return try_wait_parity(__phase_parity, return_status);
    }

#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__assert_supported_storage(); operation_status __result;
       const ::cuda::std::chrono::high_resolution_clock::time_point __start =
         ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         const auto __ptx_result                 = ::cuda::__detail::__mbarrier_try_wait_parity_primary_status(
           __native_handle(), __phase_parity_value(__phase_parity), __wait_nsec);
         __result  = __make_operation_status(__ptx_result);
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 930
#if _CCCL_CUDA_COMPILATION()
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (__assert_supported_storage(); operation_status __result;
       const ::cuda::std::chrono::high_resolution_clock::time_point __start =
         ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         __result  = __test_wait_parity_completion_only(__phase_parity);
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#endif // _CCCL_CUDA_COMPILATION()

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_parity_for(bool __phase_parity, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur) const
  {
    const operation_status __status = try_wait_parity_for(__phase_parity, __dur, return_status);
    return __completed_without_report(__status);
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_parity_for(
    bool __phase_parity, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const operation_status __status = try_wait_parity_for(__phase_parity, __dur, return_status);
    return __completed_ignoring_report(__status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_parity_until(
    bool __phase_parity, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, return_status_t) const
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()), return_status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_parity_until(bool __phase_parity, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time) const
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()));
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_parity_until(
    bool __phase_parity, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, ignore_status_t) const
  {
    return try_wait_parity_for(__phase_parity, (__time - _Clock::now()), ignore_status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t max(shared_barrier_kind __kind) noexcept
  {
    return __max_for_kind(__kind);
  }
};

namespace __detail
{
struct __issue_and_wait_access
{
  [[nodiscard]] _CCCL_DEVICE_API static bool
  __should_reissue_reported_operation(const shared_barrier::operation_status& __status, status_source __source) noexcept
  {
    return __status.__should_reissue_reported_operation(__source);
  }
};
} // namespace __detail

template <class _Issue>
[[nodiscard]] _CCCL_HOST_DEVICE_API shared_barrier::operation_status
issue_and_wait(const shared_barrier& __barrier, _Issue&& __issue, status_source __source)
{
  NV_IF_TARGET(
    NV_IS_DEVICE, (while (true) {
      shared_barrier::operation_status __result = __barrier.wait(__issue(), return_status);
      if (!::cuda::__detail::__issue_and_wait_access::__should_reissue_reported_operation(__result, __source))
      {
        return __result;
      }
    }))

  return __barrier.wait(__issue(), return_status);
}

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b)
{
  return &__b.__barrier_;
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_SHARED_BARRIER_H
