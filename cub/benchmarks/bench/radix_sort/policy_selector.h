// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_radix_sort.cuh>

#if !TUNE_BASE
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector
{
  using DominantT = cuda::std::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> ::cub::RadixSortPolicy
  {
    const auto onesweep = [] {
      const auto scaled =
        cub::detail::scale_reg_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, sizeof(DominantT));
      return cub::RadixSortOnesweepPolicy{
        scaled.threads_per_block,
        scaled.items_per_thread,
        cub::RADIX_SORT_STORE_DIRECT,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE,
        1,
        TUNE_RADIX_BITS};
    }();

    // These kernels are launched once, no point in tuning at the moment
    const auto histogram = cub::RadixSortHistogramPolicy{
      128, 16, cub::detail::radix_sort::__scale_num_parts(1, sizeof(KeyT)), TUNE_RADIX_BITS};
    const auto exclusive_sum = cub::RadixSortExclusiveSumPolicy{256, TUNE_RADIX_BITS};

    const auto scan = [] {
      const auto scaled = cub::detail::scale_mem_bound(512, 23, sizeof(OffsetT));
      return scan{scaled.threads_per_block,
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
      return cub::RadixSortDownsweepPolicy{
        scaled.threads_per_block,
        scaled.items_per_thread,
        cub::BLOCK_LOAD_DIRECT,
        cub::LOAD_LDG,
        cub::RADIX_RANK_MEMOIZE,
        cub::BLOCK_SCAN_WARP_SCANS,
        single_tile_radix_bits,
      };
    }();

    return cub::RadixSortPolicy{
      cub::RadixSortAlgorithm::onesweep,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      /* downsweep */ {},
      /* alt_downsweep */ {},
      /* upsweep */ {},
      /* alt_upsweep */ {},
      single_tile};
  }
};

template <typename KeyT, typename ValueT, typename OffsetT, cub::SortOrder SortOrder>
constexpr std::size_t max_onesweep_temp_storage_size()
{
  using portion_offset = int;

  constexpr auto active_policy = policy_selector<KeyT, ValueT, OffsetT>{}(cuda::compute_capability{});

  constexpr auto onesweep = active_policy.onesweep;
  using onesweep_policy_t = cub::detail::agent_radix_sort_onesweep_policy<
    0,
    0,
    void,
    onesweep.rank_num_private_partitions,
    onesweep.rank_algorithm,
    onesweep.scan_algorithm,
    onesweep.store_algorithm,
    onesweep.radix_bits,
    cub::NoScaling<onesweep.threads_per_block, onesweep.items_per_thread>>;

  using agent_radix_sort_onesweep_t =
    cub::AgentRadixSortOnesweep<onesweep_policy_t, SortOrder, KeyT, ValueT, OffsetT, portion_offset>;

  constexpr auto histogram = active_policy.histogram;
  using histogram_policy_t = cub::detail::agent_radix_sort_histogram_policy<
    histogram.threads_per_block,
    histogram.items_per_thread,
    histogram.num_private_partitions,
    void,
    histogram.radix_bits>;
  using hist_agent = cub::AgentRadixSortHistogram<histogram_policy_t, SortOrder, KeyT, OffsetT>;

  return cuda::std::max(sizeof(typename agent_radix_sort_onesweep_t::TempStorage),
                        sizeof(typename hist_agent::TempStorage));
}

template <typename KeyT, typename ValueT, typename OffsetT, cub::SortOrder SortOrder>
constexpr std::size_t max_temp_storage_size()
{
  using offset_t               = cub::detail::choose_offset_t<OffsetT>;
  constexpr auto active_policy = policy_selector<KeyT, ValueT, offset_t>{}(cuda::compute_capability{});
  static_assert(active_policy.algorithm == cub::RadixSortAlgorithm::onesweep);
  return max_onesweep_temp_storage_size<KeyT, ValueT, offset_t, SortOrder>();
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
