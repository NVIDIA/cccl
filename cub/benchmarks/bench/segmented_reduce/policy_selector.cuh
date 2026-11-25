// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_reduce.cuh>

// The including benchmark source must define the TUNE_* parameters via %RANGE% comments.

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> ::cub::SegmentedReducePolicy
  {
    constexpr int accum_size = int{sizeof(AccumT)};

    const auto [l_items, l_threads] =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD, accum_size);
    const auto s_items =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD, accum_size)
        .items_per_thread;
    const auto m_items =
      cub::detail::scale_mem_bound(TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK, TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD, accum_size)
        .items_per_thread;

    const auto rp = cub::ReducePassPolicy{
      l_threads, l_items, 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_LDG};
    return {rp,
            cub::SegmentedReduceWarpReducePolicy{
              rp.threads_per_block, TUNE_M_THREADS_PER_WARP, m_items, rp.vec_size, rp.load_modifier},
            cub::SegmentedReduceWarpReducePolicy{
              rp.threads_per_block, TUNE_S_THREADS_PER_WARP, s_items, rp.vec_size, rp.load_modifier}};
  }
};
#endif // !TUNE_BASE
