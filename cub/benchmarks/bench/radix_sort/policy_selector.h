// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_radix_sort.cuh>

// %//RANGE//% TUNE_RADIX_BITS bits 8:9:1
#define TUNE_RADIX_BITS 8

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#if !TUNE_BASE
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector
{
  using DominantT = cuda::std::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> ::cub::detail::radix_sort::radix_sort_policy
  {
    const auto onesweep = [] {
      const auto scaled =
        cub::detail::scale_reg_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, sizeof(DominantT));
      return radix_sort_onesweep_policy{
        scaled.block_threads,
        scaled.items_per_thread,
        1,
        ONESWEEP_RADIX_BITS,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE,
        cub::RADIX_SORT_STORE_DIRECT};
    }();

    // These kernels are launched once, no point in tuning at the moment
    const auto histogram = radix_sort_histogram_policy{
      128, 16, cub::detail::radix_sort::__scale_num_parts(1, sizeof(KeyT)), ONESWEEP_RADIX_BITS};
    const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, ONESWEEP_RADIX_BITS};

    const auto scan = [] {
      const auto scaled = cub::detail::scale_mem_bound(512, 23, sizeof(OffsetT));
      return scan{scaled.block_threads,
                  scaled.items_per_thread,
                  cub::BLOCK_LOAD_WARP_TRANSPOSE,
                  cub::LOAD_DEFAULT,
                  cub::BLOCK_STORE_WARP_TRANSPOSE,
                  cub::BLOCK_SCAN_RAKING_MEMOIZE};
    }();

    // No point in tuning
    const int single_tile_radix_bits = (sizeof(KeyT) > 1) ? 6 : 5;

    // No point in tuning single-tile policy
    const auto single_tile = [] {
      const auto scaled = cub::detail::scale_reg_bound(256, 19, sizeof(DominantT));
      return cub::detail::radix_sort::radix_sort_downsweep_policy{
        scaled.block_threads,
        scaled.items_per_thread,
        single_tile_radix_bits,
        cub::BLOCK_LOAD_DIRECT,
        cub::LOAD_LDG,
        cub::RADIX_RANK_MEMOIZE,
        cub::BLOCK_SCAN_WARP_SCANS,
      };
    }();

    return radix_sort_policy{
      /* use_onesweep */ true,
      /* onesweep_radix_bits */ TUNE_RADIX_BITS,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      /* downsweep */ {},
      /* alt_downsweep */ {},
      /* upsweep */ {},
      /* alt_upsweep */ {},
      single_tile,
      /* segmented not used */ {},
      /* alt_segmented not used */ {}};
  }
};

template <typename KeyT, typename ValueT, typename OffsetT, cub::SortOrder SortOrder>
constexpr std::size_t max_onesweep_temp_storage_size()
{
  using portion_offset  = int;
  using onesweep_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::OnesweepPolicy;
  using agent_radix_sort_onesweep_t =
    cub::AgentRadixSortOnesweep<onesweep_policy, SortOrder, KeyT, ValueT, OffsetT, portion_offset>;

  using hist_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::HistogramPolicy;
  using hist_agent  = cub::AgentRadixSortHistogram<hist_policy, SortOrder, KeyT, OffsetT>;

  return cuda::std::max(sizeof(typename agent_radix_sort_onesweep_t::TempStorage),
                        sizeof(typename hist_agent::TempStorage));
}

template <typename KeyT, typename ValueT, typename OffsetT, cub::SortOrder SortOrder>
constexpr std::size_t max_temp_storage_size()
{
  using policy_t = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t;

  static_assert(policy_t::ONESWEEP);
  return max_onesweep_temp_storage_size<KeyT, ValueT, OffsetT, SortOrder>();
}

template <typename KeyT, typename ValueT, typename OffsetT, cub::SortOrder SortOrder>
constexpr bool fits_in_default_shared_memory()
{
  return max_temp_storage_size<KeyT, ValueT, OffsetT, SortOrder>() < cub::detail::max_smem_per_block;
}
#else // TUNE_BASE
template <typename, typename, typename, auto>
constexpr bool fits_in_default_shared_memory()
{
  return true;
}
#endif // TUNE_BASE
