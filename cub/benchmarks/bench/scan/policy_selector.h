// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

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
