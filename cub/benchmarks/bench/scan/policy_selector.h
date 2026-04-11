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
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> cub::detail::scan::scan_policy
  {
#  if USES_WARPSPEED()
    return {cub::detail::scan::scan_algorithm::warpspeed,
            cub::detail::scan::scan_lookback_policy{},
            cub::detail::scan::scan_warpspeed_policy{
              TUNE_NUM_REDUCE_SCAN_WARPS, TUNE_NUM_LOOKBACK_ITEMS, TUNE_ITEMS_PLUS_ONE - 1}};
#  else
    return cub::detail::scan::make_mem_scaled_lookback_scan_policy(
      TUNE_THREADS,
      TUNE_ITEMS,
      int{sizeof(AccumT)},
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_STORE_ALGORITHM,
      cub::BLOCK_SCAN_WARP_SCANS,
      delay_constructor_policy);
#  endif
  }
};
#endif // !TUNE_BASE
