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

#include <cub/util_debug.cuh>

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
struct DeviceDescription
{
  ::cuda::std::optional<::cuda::compute_capability> __cc_{};
  ::cuda::std::optional<::cuda::std::uint32_t> __max_sm_count_{};
};

struct __get_runs_on_t;

// A guarantee that names the device an algorithm runs on. An algorithm uses it in place of a
// runtime device query, so it can select a backend, pick tuning parameters and compute
// temporary storage requirements without a device being present. The same guarantee must be
// passed to the temporary storage query and to the later call that does the work.
class RunsOn : public ::cuda::execution::__guarantee
{
public:
  _CCCL_HIDE_FROM_ABI constexpr RunsOn() noexcept = default;

  _CCCL_API explicit constexpr RunsOn(DeviceDescription __descr) noexcept
      : __description_{::cuda::std::move(__descr)}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class LauncherFactory>
  [[nodiscard]] _CCCL_API constexpr ::cudaError_t compute_capability(
    const LauncherFactory& __launcher_factory, const void* d_temp_storage, ::cuda::compute_capability& __ret) const
  {
    if (const auto& __cc = description().__cc_; __cc.has_value())
    {
      __ret = *__cc;
#ifdef CCCL_ENABLE_ASSERTIONS
      // We only check this invariant during the "run" phase of a CUB algorithm because the
      // user is allowed to do temporary storage requirement calculations on a host with a
      // different GPU or no GPU at all.
      //
      // But once they go to execute the algorithm, not only must they have a GPU (obviously),
      // but that GPU should be *exactly* what they told us it would be.
      if (d_temp_storage)
      {
        ::cuda::compute_capability __actual_cc{};

        if (const auto __err = CubDebug(__launcher_factory.PtxComputeCap(__actual_cc)))
        {
          return __err;
        }

        if (__actual_cc != __ret)
        {
          return ::cudaErrorInvalidValue;
        }
      }
#else // ^^^ assertions ^^^ / vvv no assertions vvv
      static_cast<void>(d_temp_storage);
#endif // ^^^ no assertions ^^^
    }
    else if (const auto __err = CubDebug(__launcher_factory.PtxComputeCap(__ret)))
    {
      return __err;
    }

    return ::cudaSuccess;
  }

  [[nodiscard]] _CCCL_API constexpr const DeviceDescription& description() const noexcept
  {
    return __description_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr const RunsOn& query(const __get_runs_on_t&) const noexcept
  {
    return *this;
  }

private:
  DeviceDescription __description_{};
};

struct __get_runs_on_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_runs_on_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr decltype(auto) _CCCL_STATIC_CALL_OPERATOR(const _Env& __env) noexcept
  {
    static_assert(noexcept(__env.query(__get_runs_on_t{})), "The RunsOn guarantee must be queryable without throwing");
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
