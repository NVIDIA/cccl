// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_scan.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
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
      cub::detail::delay_constructor_policy_from_type<delay_constructor_t>);
  }
};
#endif // !TUNE_BASE
