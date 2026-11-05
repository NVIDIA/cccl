// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/round_down.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <int LogicalWarpSize>
inline constexpr bool is_valid_logical_warp_size_v = LogicalWarpSize >= 1 && LogicalWarpSize <= warp_threads;

//! @brief Returns the calling thread's lane index within its logical warp.
//! @pre For non-power-of-two sizes, only the first LogicalWarpSize lanes of each hardware warp may call this function.
template <int LogicalWarpSize = warp_threads>
[[nodiscard]] _CCCL_DEVICE_API
_CCCL_FORCEINLINE int logical_lane_id(::cuda::std::integral_constant<int, LogicalWarpSize> = {})
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  // _CCCL_ASSUME interferes with constant propagation
  if constexpr (LogicalWarpSize == 1)
  {
    return 0;
  }
  else
  {
    constexpr bool is_full_warp           = LogicalWarpSize == warp_threads;
    constexpr auto is_single_logical_warp = is_full_warp || !::cuda::is_power_of_two(LogicalWarpSize);
    constexpr auto lane_limit             = is_single_logical_warp ? warp_threads : LogicalWarpSize;

    const auto lane             = ::cuda::ptx::get_sreg_laneid();
    const auto logical_lane     = is_single_logical_warp ? lane : lane % LogicalWarpSize;
    const auto logical_lane_int = static_cast<int>(logical_lane);
    _CCCL_ASSUME(logical_lane_int < lane_limit);
    // TODO(fbusato): Re-enable the nonnegative hint after updating guarded load/store bounds checks to avoid redundant
    // addresses. _CCCL_ASSUME(logical_lane_int >= 0);
    return logical_lane_int;
  }
}

//! @brief Returns the calling thread's logical warp index within its hardware warp.
template <int LogicalWarpSize = warp_threads>
[[nodiscard]] _CCCL_DEVICE_API
_CCCL_FORCEINLINE int logical_warp_id(::cuda::std::integral_constant<int, LogicalWarpSize> = {})
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  constexpr bool is_full_warp           = LogicalWarpSize == warp_threads;
  constexpr auto is_single_logical_warp = is_full_warp || !::cuda::is_power_of_two(LogicalWarpSize);
  // _CCCL_ASSUME interferes with constant propagation
  if constexpr (is_single_logical_warp)
  {
    return 0;
  }
  else
  {
    const auto lane            = ::cuda::ptx::get_sreg_laneid();
    const auto logical_warp_id = static_cast<int>(lane / LogicalWarpSize);
    _CCCL_ASSUME(logical_warp_id >= 0 && logical_warp_id < warp_threads / LogicalWarpSize);
    return logical_warp_id;
  }
}

//! @brief Returns the calling thread's logical warp base ID (id * warp_size) within its hardware warp.
template <int LogicalWarpSize = warp_threads>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int
logical_warp_base_id(::cuda::std::integral_constant<int, LogicalWarpSize> logical_warp_size = {})
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  constexpr bool is_full_warp           = LogicalWarpSize == warp_threads;
  constexpr auto is_single_logical_warp = is_full_warp || !::cuda::is_power_of_two(LogicalWarpSize);
  // _CCCL_ASSUME interferes with constant propagation
  if constexpr (is_single_logical_warp)
  {
    return 0;
  }
  else
  {
    const auto warp_base_id = CUB_NS_QUALIFIER::detail::logical_warp_id(logical_warp_size) * LogicalWarpSize;
    _CCCL_ASSUME(warp_base_id >= 0 && warp_base_id < warp_threads);
    return warp_base_id;
  }
}

//! @brief Returns the calling thread's logical warp member mask within its hardware warp.
//! @pre For non-power-of-two sizes, only the first LogicalWarpSize lanes of each hardware warp may call this function.
template <int LogicalWarpSize = warp_threads>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE uint32_t warp_mask()
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  constexpr uint32_t member_mask = 0xFFFFFFFFu >> (warp_threads - LogicalWarpSize);
  return member_mask << CUB_NS_QUALIFIER::detail::logical_warp_base_id<LogicalWarpSize>();
}
} // namespace detail
CUB_NAMESPACE_END
