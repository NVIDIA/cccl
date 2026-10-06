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
#include <cuda/std/__algorithm/min.h>
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
class DeviceDescription
{
public:
  _CCCL_HIDE_FROM_ABI constexpr DeviceDescription() = default;

  _CCCL_API explicit DeviceDescription(::cuda::compute_capability cc, ::cuda::std::uint32_t max_sm_count)
      : __cc_{::cuda::std::move(cc)}
      , __max_sm_count_{max_sm_count}
  {}

  [[nodiscard]] _CCCL_API ::cuda::compute_capability __compute_capability() const noexcept
  {
    return __cc_;
  }

  [[nodiscard]] _CCCL_API ::cuda::std::uint32_t __max_sm_count() const noexcept
  {
    return __max_sm_count_;
  }

private:
  ::cuda::compute_capability __cc_{};
  ::cuda::std::uint32_t __max_sm_count_{};
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

  _CCCL_API constexpr explicit RunsOn(DeviceDescription __descr) noexcept
      : __description_{::cuda::std::move(__descr)}
  {}

  [[nodiscard]] _CCCL_API constexpr const ::cuda::std::optional<DeviceDescription>& description() const noexcept
  {
    return __description_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class LauncherFactory>
  [[nodiscard]] _CCCL_API ::cudaError_t concrete_description(
    const LauncherFactory& launcher_factory, const void* d_temp_storage, DeviceDescription& descr) const
  {
    if (__description_.has_value())
    {
      descr = *__description_;
#ifdef CCCL_ENABLE_ASSERTIONS
      // We only check this invariant during the "run" phase of a CUB algorithm because the
      // user is allowed to do temporary storage requirement calculations on a host with a
      // different GPU or no GPU at all.
      //
      // But once they go to execute the algorithm, not only must they have a GPU (obviously),
      // but that GPU should be *exactly* what they told us it would be.
      if (d_temp_storage)
      {
        ::cuda::compute_capability actual_cc{};

        if (const auto error = CubDebug(launcher_factory.PtxComputeCap(actual_cc)))
        {
          return error;
        }

        if (actual_cc != descr.__compute_capability())
        {
          return ::cudaErrorInvalidValue;
        }
      }
#else // ^^^ assertions ^^^ / vvv no assertions vvv
      static_cast<void>(d_temp_storage);
#endif // ^^^ no assertions ^^^
      return ::cudaSuccess;
    }

    ::compute_capability cc{};

    if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
    {
      return error;
    }

    int device_ordinal{};

    if (const auto error = CubDebug(::cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    int sm_count{};

    if (const auto error =
          CubDebug(::cudaDeviceGetAttribute(&sm_count, ::cudaDevAttrMultiProcessorCount, device_ordinal)))
    {
      return error;
    }

    descr = DeviceDescription{cc, sm_count};
    return ::cudaSuccess;
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API constexpr const RunsOn& query(const __get_runs_on_t&) const noexcept
  {
    return *this;
  }

private:
  ::cuda::std::optional<DeviceDescription> __description_{};
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
