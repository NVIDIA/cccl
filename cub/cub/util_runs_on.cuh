// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/guarantee.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

CUB_NAMESPACE_BEGIN

namespace experimental
{
struct device_description
{
  ::cuda::std::optional<::cuda::compute_capability> __cc_{};
  ::cuda::std::optional<::cuda::std::uint32_t> __max_sm_count_{};
};

struct __get_runs_on_t;

// A guarantee that names the device an algorithm runs on. An algorithm uses it in place of a
// runtime device query, so it can select a backend, pick tuning parameters and compute
// temporary storage requirements without a device being present. The same guarantee must be
// passed to the temporary storage query and to the later call that does the work.
class runs_on : public ::cuda::execution::__guarantee
{
public:
  _CCCL_HIDE_FROM_ABI constexpr runs_on() noexcept = default;

  _CCCL_API explicit constexpr runs_on(device_description __descr) noexcept
      : __description_{::cuda::std::move(__descr)}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _LauncherFactory>
  [[nodiscard]] _CCCL_API constexpr ::cuda::compute_capability
  compute_capability(const _LauncherFactory& __launcher_factory) const
  {
    ::cuda::compute_capability __ret{};

    if (auto&& __descr = description(); __descr.__cc_.has_value())
    {
      __ret = *__descr.__cc_;
#ifdef CCCL_ENABLE_ASSERTIONS
      ::cuda::compute_capability __actual_cc{};

      _CCCL_TRY_RUNTIME_API(__launcher_factory.PtxComputeCap, "PtxComputeCap failed", __actual_cc);
      if (__actual_cc != __ret)
      {
        _CCCL_THROW(::std::invalid_argument,
                    "runs_on compute_capability does not match current device compute capability");
      }
#endif // CCCL_ENABLE_ASSERTIONS
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(__launcher_factory.PtxComputeCap, "PtxComputeCap failed", __ret);
    }

    return __ret;
  }

  [[nodiscard]] _CCCL_API constexpr const device_description& description() const noexcept
  {
    return __description_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr const runs_on& query(const __get_runs_on_t&) const noexcept
  {
    return *this;
  }

private:
  device_description __description_{};
};

struct __get_runs_on_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_runs_on_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr decltype(auto) _CCCL_STATIC_CALL_OPERATOR(const _Env& __env) noexcept
  {
    static_assert(noexcept(__env.query(__get_runs_on_t{})), "The runs_on guarantee must be queryable without throwing");
    return __env.query(__get_runs_on_t{});
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr bool query(::cuda::std::execution::forwarding_query_t) noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_runs_on = __get_runs_on_t{};
} // namespace experimental

CUB_NAMESPACE_END

#include <cuda/std/__cccl/epilogue.h>
