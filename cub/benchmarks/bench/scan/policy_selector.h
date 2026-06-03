// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#ifndef USES_WARPSPEED
#  define USES_WARPSPEED() 0
#endif

#if !TUNE_BASE
#  if !USES_WARPSPEED()
#    include <look_back_helper.cuh>
#  endif // !USES_WARPSPEED()

template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::ScanPolicy
  {
#  if USES_WARPSPEED()
    return {cub::ScanAlgorithm::warpspeed,
            cub::ScanLookbackPolicy{},
            cub::ScanWarpspeedPolicy{
              TUNE_NUM_REDUCE_SCAN_WARPS,
              TUNE_NUM_LOOKBACK_ITEMS,
              TUNE_ITEMS_PLUS_ONE - 1,
              TUNE_LOOKBACK_STAGES,
              TUNE_BLOCK_IDX_STAGES}};
#  else
    return cub::detail::scan::make_mem_scaled_lookback_scan_policy(
      TUNE_THREADS,
      TUNE_ITEMS,
      int{sizeof(AccumT)},
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_STORE_ALGORITHM,
      cub::BLOCK_SCAN_WARP_SCANS,
      lookback_delay_policy);
#  endif
  }
};
#endif // !TUNE_BASE
