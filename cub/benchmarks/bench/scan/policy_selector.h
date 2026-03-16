// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#if !TUNE_BASE
#  include "look_back_helper.cuh"
#endif // !TUNE_BASE

#ifndef USES_WARPSPEED
#  define USES_WARPSPEED() 0
#endif

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
#  if USES_WARPSPEED()
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> cub::detail::scan::scan_policy
  {
    static constexpr int num_reduce_and_scan_warps = TUNE_NUM_REDUCE_SCAN_WARPS;
    static constexpr int num_look_ahead_items      = TUNE_NUM_LOOKBACK_ITEMS;
    static constexpr int items_per_thread          = TUNE_ITEMS_PLUS_ONE - 1;

    static constexpr int num_threads_per_warp = 32;
    static constexpr int num_load_warps       = 1;
    static constexpr int num_sched_warps      = 1;
    static constexpr int num_look_ahead_warps = 1;

    static constexpr int num_total_warps =
      2 * num_reduce_and_scan_warps + num_load_warps + num_sched_warps + num_look_ahead_warps;
    static constexpr int num_total_threads    = num_total_warps * num_threads_per_warp;
    static constexpr int squad_reduce_threads = num_reduce_and_scan_warps * num_threads_per_warp;
    static constexpr int tile_size            = items_per_thread * squad_reduce_threads;

    auto warpspeed_policy = cub::detail::scan::scan_warpspeed_policy{
      true,
      num_reduce_and_scan_warps,
      num_reduce_and_scan_warps,
      num_load_warps,
      num_sched_warps,
      num_look_ahead_warps,
      num_look_ahead_items,
      num_total_threads,
      items_per_thread,
      tile_size};

    return cub::detail::scan::scan_policy{
      num_total_threads,
      items_per_thread,
      cub::BLOCK_LOAD_WARP_TRANSPOSE,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_WARP_TRANSPOSE,
      cub::BLOCK_SCAN_WARP_SCANS,
      cub::detail::delay_constructor_policy{cub::detail::delay_constructor_kind::fixed_delay, 350, 450},
      warpspeed_policy};
  }
#  else
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> cub::detail::scan::scan_policy
  {
    return cub::detail::scan::make_mem_scaled_scan_policy(
      TUNE_THREADS,
      TUNE_ITEMS,
      int{sizeof(AccumT)},
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_STORE_ALGORITHM,
      cub::BLOCK_SCAN_WARP_SCANS,
      delay_constructor_policy);
  }
#  endif
};
#endif // !TUNE_BASE
