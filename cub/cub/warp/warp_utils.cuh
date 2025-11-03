// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <int LogicalWarpSize>
inline constexpr bool is_valid_logical_warp_size_v = LogicalWarpSize >= 1 && LogicalWarpSize <= detail::warp_threads;

template <int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE
_CCCL_FORCEINLINE int logical_lane_id(::cuda::std::integral_constant<int, LogicalWarpSize> = {})
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  auto lane                             = ::cuda::ptx::get_sreg_laneid();
  constexpr bool is_full_warp           = LogicalWarpSize == detail::warp_threads;
  constexpr auto is_single_logical_warp = is_full_warp || !::cuda::is_power_of_two(LogicalWarpSize);
  auto logical_lane =
    static_cast<int>(is_single_logical_warp ? lane : (LogicalWarpSize == 1 ? 0 : lane % LogicalWarpSize));
  _CCCL_ASSUME(logical_lane >= 0 && logical_lane < LogicalWarpSize);
  return logical_lane;
}

template <int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE
_CCCL_FORCEINLINE int logical_warp_id(::cuda::std::integral_constant<int, LogicalWarpSize> = {})
{
  static_assert(is_valid_logical_warp_size_v<LogicalWarpSize>, "invalid logical warp size");
  auto lane                             = ::cuda::ptx::get_sreg_laneid();
  constexpr bool is_full_warp           = LogicalWarpSize == detail::warp_threads;
  constexpr auto is_single_logical_warp = is_full_warp || !::cuda::is_power_of_two(LogicalWarpSize);
  auto logical_warp_id                  = static_cast<int>(is_single_logical_warp ? 0 : lane / LogicalWarpSize);
  _CCCL_ASSUME(logical_warp_id >= 0 && logical_warp_id < detail::warp_threads / LogicalWarpSize);
  return logical_warp_id;
}

template <int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int
logical_warp_base_id(::cuda::std::integral_constant<int, LogicalWarpSize> logical_warp_size = {})
{
  return cub::detail::logical_warp_id(logical_warp_size) * LogicalWarpSize;
}
} // namespace detail
CUB_NAMESPACE_END
