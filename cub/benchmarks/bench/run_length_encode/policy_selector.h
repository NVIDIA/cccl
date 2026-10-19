// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_run_length_encode.cuh>

#ifndef USES_LOOKAHEAD
#  define USES_LOOKAHEAD() 0
#endif

#if !TUNE_BASE
#  if !USES_LOOKAHEAD()
#    include <look_back_helper.cuh>
#  endif // !USES_LOOKAHEAD()

struct bench_encode_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::RleEncodePolicy
  {
#  if USES_LOOKAHEAD()
    // the lookback policy is never used: the tuning build targets a single sm_100+ architecture, where every
    // benched type is lookahead-viable, so the streaming fallback is not instantiated
    return {cub::RleAlgorithm::lookahead,
            cub::RleLookbackPolicy{},
            cub::RleLookaheadPolicy{
              TUNE_LA_IPT, TUNE_LA_CW, TUNE_LA_KRS, TUNE_LA_PRS, TUNE_LA_PLL, TUNE_LA_DPLL, TUNE_LA_DMRT, TUNE_LA_FST}};
#  else // USES_LOOKAHEAD()
    return {
      cub::RleAlgorithm::lookback,
      {TUNE_THREADS,
       TUNE_ITEMS,
       TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
       TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : cub::LOAD_CA,
       cub::BLOCK_SCAN_WARP_SCANS,
       lookback_delay_policy},
      {},
    };
#  endif // USES_LOOKAHEAD()
  }
};
#endif // !TUNE_BASE
