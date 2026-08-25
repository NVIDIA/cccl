//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_RUNS_ON_H
#define __CUDA___EXECUTION_RUNS_ON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/guarantee.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

struct device_description
{
  static constexpr ::cuda::std::uint32_t unspecified = static_cast<::cuda::std::uint32_t>(-1);

  ::cuda::std::uint32_t __max_sms_ = unspecified;

  [[nodiscard]] _CCCL_API constexpr bool __has_max_sms() const noexcept
  {
    return __max_sms_ != unspecified;
  }
};

struct __get_runs_on_t;

// A guarantee that names the device an algorithm runs on. An algorithm uses it in place of a runtime device query, so
// it can select a backend, pick tuning parameters and compute temporary storage requirements without a device being
// present. The same guarantee must be passed to the temporary storage query and to the later call that does the work.
class runs_on : public __guarantee
{
public:
  _CCCL_API explicit constexpr runs_on(::cuda::compute_capability __cc) noexcept
      : runs_on{__cc, {}}
  {}

  _CCCL_API constexpr runs_on(::cuda::compute_capability __cc, device_description __description) noexcept
      : __cc_{__cc}
      , __description_{__description}
  {}

  [[nodiscard]] _CCCL_API constexpr ::cuda::compute_capability compute_capability() const noexcept
  {
    return __cc_;
  }

  [[nodiscard]] _CCCL_API constexpr device_description description() const noexcept
  {
    return __description_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr runs_on query(const __get_runs_on_t&) const noexcept
  {
    return *this;
  }

private:
  ::cuda::compute_capability __cc_;
  device_description __description_{};
};

struct __get_runs_on_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_runs_on_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "The runs_on guarantee must be queryable without throwing");
    return __env.query(*this);
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_runs_on = __get_runs_on_t{};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_RUNS_ON_H
