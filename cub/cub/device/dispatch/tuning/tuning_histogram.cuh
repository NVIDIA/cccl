// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

enum class HistogramHighBinAlgorithm
{
  global_memory_privatized,
  cooperative
};

enum class HistogramCacheAlgorithm
{
  none,
  single_probe,
  cuckoo
};

enum class HistogramSpillAlgorithm
{
  output,
  global_memory_privatized
};

enum class HistogramAggregationAlgorithm
{
  direct,
  warp_coalesced,
  rle
};

namespace detail::histogram
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(HistogramHighBinAlgorithm value) noexcept
{
  return value == HistogramHighBinAlgorithm::cooperative
         ? "HistogramHighBinAlgorithm::cooperative"
         : "HistogramHighBinAlgorithm::global_memory_privatized";
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(HistogramCacheAlgorithm value) noexcept
{
  return value == HistogramCacheAlgorithm::none ? "HistogramCacheAlgorithm::none"
       : value == HistogramCacheAlgorithm::single_probe
         ? "HistogramCacheAlgorithm::single_probe"
         : "HistogramCacheAlgorithm::cuckoo";
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(HistogramSpillAlgorithm value) noexcept
{
  return value == HistogramSpillAlgorithm::output
         ? "HistogramSpillAlgorithm::output"
         : "HistogramSpillAlgorithm::global_memory_privatized";
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(HistogramAggregationAlgorithm value) noexcept
{
  return value == HistogramAggregationAlgorithm::direct ? "HistogramAggregationAlgorithm::direct"
       : value == HistogramAggregationAlgorithm::warp_coalesced
         ? "HistogramAggregationAlgorithm::warp_coalesced"
         : "HistogramAggregationAlgorithm::rle";
}
} // namespace detail::histogram

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, HistogramHighBinAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}
inline ::std::ostream& operator<<(::std::ostream& os, HistogramCacheAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}
inline ::std::ostream& operator<<(::std::ostream& os, HistogramSpillAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}
inline ::std::ostream& operator<<(::std::ostream& os, HistogramAggregationAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}
#endif

//! The tuning policy for one DeviceHistogram privatization technique.
struct HistogramPrivatizationPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  int vec_size; //!< Vectorization size for loading samples
  BlockLoadAlgorithm load_algorithm; //!< Algorithm used for loading samples
  CacheLoadModifier load_modifier; //!< Cache modifier used for loading samples
  bool rle_compress; //!< Whether to locally run-length encode samples
  bool work_stealing; //!< Whether blocks dequeue tiles from a global work queue

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPrivatizationPolicy& lhs, const HistogramPrivatizationPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.work_stealing == rhs.work_stealing;
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const HistogramPrivatizationPolicy& p)
  {
    return os
        << "HistogramPrivatizationPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .vec_size = " << p.vec_size
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .rle_compress = " << p.rle_compress << ", .work_stealing = " << p.work_stealing << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all DeviceHistogram sweep passes.
struct HistogramPolicy
{
  HistogramPrivatizationPolicy gmem; //!< Policy for global-memory privatization
  HistogramPrivatizationPolicy static_smem; //!< Policy for compile-time-sized shared-memory privatization
  HistogramPrivatizationPolicy dynamic_smem; //!< Policy for runtime-sized shared-memory privatization
  int init_threads_per_block; //!< Number of threads in a histogram initialization block
  int max_privatized_static_smem_single_channel_bytes; //!< Single-channel compile-time-sized SMEM limit
  int max_privatized_dynamic_smem_single_channel_bytes; //!< Single-channel runtime-sized SMEM limit
  int static_smem_min_blocks_per_sm; //!< Minimum blocks per SM requested by the static-SMEM launch bounds
  int max_privatized_dynamic_smem_multi_channel_range_bytes; //!< Multi-channel HistogramRange SMEM limit
  int max_privatized_dynamic_smem_2_channel_even_bytes; //!< Two-channel HistogramEven SMEM limit
  int max_privatized_dynamic_smem_3_channel_even_bytes; //!< Three-channel HistogramEven SMEM limit
  int max_privatized_dynamic_smem_4_channel_even_bytes; //!< Four-channel HistogramEven SMEM limit
  int max_output_histogram_bytes_for_init_kernel_pdl; //!< Largest output allocation for init-kernel PDL
  HistogramHighBinAlgorithm high_bin_algorithm       = HistogramHighBinAlgorithm::global_memory_privatized;
  HistogramCacheAlgorithm high_bin_cache             = HistogramCacheAlgorithm::single_probe;
  HistogramSpillAlgorithm high_bin_spill             = HistogramSpillAlgorithm::global_memory_privatized;
  HistogramAggregationAlgorithm high_bin_aggregation = HistogramAggregationAlgorithm::rle;
  int high_bin_cache_bytes_per_channel               = 16384;
  int high_bin_cache_count_replicas                  = 1;
  int high_bin_cache_cuckoo_max_histogram_bytes      = 1048576;
  int high_bin_items_per_thread                      = 4;
  int high_bin_threads_per_block                     = 0; //!< High-bin block size; 0 inherits threads_per_block
  int high_bin_interpolation_min_level_bytes         = 2052;
  int high_bin_min_histogram_bytes                   = 0;
  //! Target resident cooperative blocks per SM. Zero keeps the occupancy-derived grid and a one-block launch bound.
  int high_bin_blocks_per_sm        = 0;
  int high_bin_grid_items_per_block = 0;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int high_bin_threads() const noexcept
  {
    return high_bin_threads_per_block != 0 ? high_bin_threads_per_block : gmem.threads_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int high_bin_min_blocks() const noexcept
  {
    return high_bin_blocks_per_sm != 0 ? high_bin_blocks_per_sm : 1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int high_bin_grid_items() const noexcept
  {
    return high_bin_grid_items_per_block != 0
           ? high_bin_grid_items_per_block
           : high_bin_threads() * high_bin_items_per_thread;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.gmem == rhs.gmem && lhs.static_smem == rhs.static_smem && lhs.dynamic_smem == rhs.dynamic_smem
        && lhs.init_threads_per_block == rhs.init_threads_per_block
        && lhs.max_privatized_static_smem_single_channel_bytes == rhs.max_privatized_static_smem_single_channel_bytes
        && lhs.max_privatized_dynamic_smem_single_channel_bytes == rhs.max_privatized_dynamic_smem_single_channel_bytes
        && lhs.static_smem_min_blocks_per_sm == rhs.static_smem_min_blocks_per_sm
        && lhs.max_privatized_dynamic_smem_multi_channel_range_bytes
             == rhs.max_privatized_dynamic_smem_multi_channel_range_bytes
        && lhs.max_privatized_dynamic_smem_2_channel_even_bytes == rhs.max_privatized_dynamic_smem_2_channel_even_bytes
        && lhs.max_privatized_dynamic_smem_3_channel_even_bytes == rhs.max_privatized_dynamic_smem_3_channel_even_bytes
        && lhs.max_privatized_dynamic_smem_4_channel_even_bytes == rhs.max_privatized_dynamic_smem_4_channel_even_bytes
        && lhs.max_output_histogram_bytes_for_init_kernel_pdl == rhs.max_output_histogram_bytes_for_init_kernel_pdl
        && lhs.high_bin_algorithm == rhs.high_bin_algorithm && lhs.high_bin_cache == rhs.high_bin_cache
        && lhs.high_bin_spill == rhs.high_bin_spill && lhs.high_bin_aggregation == rhs.high_bin_aggregation
        && lhs.high_bin_cache_bytes_per_channel == rhs.high_bin_cache_bytes_per_channel
        && lhs.high_bin_cache_count_replicas == rhs.high_bin_cache_count_replicas
        && lhs.high_bin_cache_cuckoo_max_histogram_bytes == rhs.high_bin_cache_cuckoo_max_histogram_bytes
        && lhs.high_bin_items_per_thread == rhs.high_bin_items_per_thread
        && lhs.high_bin_threads_per_block == rhs.high_bin_threads_per_block
        && lhs.high_bin_interpolation_min_level_bytes == rhs.high_bin_interpolation_min_level_bytes
        && lhs.high_bin_min_histogram_bytes == rhs.high_bin_min_histogram_bytes
        && lhs.high_bin_blocks_per_sm == rhs.high_bin_blocks_per_sm
        && lhs.high_bin_grid_items_per_block == rhs.high_bin_grid_items_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const HistogramPolicy& p)
  {
    return os
        << "HistogramPolicy { .gmem = " << p.gmem << ", .static_smem = " << p.static_smem
        << ", .dynamic_smem = " << p.dynamic_smem << ", .init_threads_per_block = " << p.init_threads_per_block
        << ", .max_privatized_static_smem_single_channel_bytes = " << p.max_privatized_static_smem_single_channel_bytes
        << ", .max_privatized_dynamic_smem_single_channel_bytes = "
        << p.max_privatized_dynamic_smem_single_channel_bytes << ", .static_smem_min_blocks_per_sm = "
        << p.static_smem_min_blocks_per_sm << ", .max_privatized_dynamic_smem_multi_channel_range_bytes = "
        << p.max_privatized_dynamic_smem_multi_channel_range_bytes
        << ", .max_privatized_dynamic_smem_2_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_2_channel_even_bytes
        << ", .max_privatized_dynamic_smem_3_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_3_channel_even_bytes
        << ", .max_privatized_dynamic_smem_4_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_4_channel_even_bytes
        << ", .max_output_histogram_bytes_for_init_kernel_pdl = " << p.max_output_histogram_bytes_for_init_kernel_pdl
        << ", .high_bin_algorithm = " << p.high_bin_algorithm << ", .high_bin_cache = " << p.high_bin_cache
        << ", .high_bin_spill = " << p.high_bin_spill << ", .high_bin_aggregation = " << p.high_bin_aggregation
        << ", .high_bin_cache_bytes_per_channel = " << p.high_bin_cache_bytes_per_channel
        << ", .high_bin_cache_count_replicas = " << p.high_bin_cache_count_replicas
        << ", .high_bin_cache_cuckoo_max_histogram_bytes = " << p.high_bin_cache_cuckoo_max_histogram_bytes
        << ", .high_bin_items_per_thread = " << p.high_bin_items_per_thread
        << ", .high_bin_threads_per_block = " << p.high_bin_threads_per_block
        << ", .high_bin_interpolation_min_level_bytes = " << p.high_bin_interpolation_min_level_bytes
        << ", .high_bin_min_histogram_bytes = " << p.high_bin_min_histogram_bytes << ", .high_bin_blocks_per_sm = "
        << p.high_bin_blocks_per_sm << ", .high_bin_grid_items_per_block = " << p.high_bin_grid_items_per_block << " }";
  }
#endif
};

namespace detail::histogram
{
enum class privatization_mode
{
  gmem,
  static_smem,
  dynamic_smem
};

template <typename CounterT, int NumActiveChannels>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int max_privatized_smem_bins(int max_privatized_smem_bytes)
{
  static_assert(NumActiveChannels > 0);
  if (max_privatized_smem_bytes <= 0)
  {
    return 0;
  }
  return max_privatized_smem_bytes / int{sizeof(CounterT)} / NumActiveChannels;
}

template <bool IsEven, int NumActiveChannels>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int dynamic_smem_limit_bytes(const HistogramPolicy& policy)
{
  if constexpr (NumActiveChannels == 1)
  {
    return policy.max_privatized_dynamic_smem_single_channel_bytes;
  }
  else if constexpr (IsEven)
  {
    return NumActiveChannels == 2 ? policy.max_privatized_dynamic_smem_2_channel_even_bytes
         : NumActiveChannels == 3 ? policy.max_privatized_dynamic_smem_3_channel_even_bytes
         : NumActiveChannels == 4
           ? policy.max_privatized_dynamic_smem_4_channel_even_bytes
           : 0;
  }
  else
  {
    return policy.max_privatized_dynamic_smem_multi_channel_range_bytes;
  }
}

template <bool IsEven, typename CounterT, int NumActiveChannels>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
select_privatization_mode(const HistogramPolicy& policy, int num_bins) -> privatization_mode
{
  if (num_bins <= 0)
  {
    return privatization_mode::gmem;
  }

  const int static_smem_max_bins =
    max_privatized_smem_bins<CounterT, 1>(policy.max_privatized_static_smem_single_channel_bytes);
  const int dynamic_smem_max_bytes = dynamic_smem_limit_bytes<IsEven, NumActiveChannels>(policy);
  const int dynamic_smem_max_bins  = max_privatized_smem_bins<CounterT, NumActiveChannels>(dynamic_smem_max_bytes);
  if (num_bins <= static_smem_max_bins)
  {
    return privatization_mode::static_smem;
  }
  if (num_bins <= dynamic_smem_max_bins)
  {
    return privatization_mode::dynamic_smem;
  }
  return privatization_mode::gmem;
}

// The C Parallel API erases CounterT before host dispatch, so its bridge must select from the
// preserved runtime counter width. Typed CUB dispatch uses the overload above.
template <bool IsEven, int NumActiveChannels>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
select_privatization_mode_for_counter_size(const HistogramPolicy& policy, int num_bins, int counter_size_bytes)
  -> privatization_mode
{
  if (num_bins <= 0 || counter_size_bytes <= 0)
  {
    return privatization_mode::gmem;
  }

  const int static_smem_max_bins = policy.max_privatized_static_smem_single_channel_bytes / counter_size_bytes;
  const int dynamic_smem_max_bins =
    dynamic_smem_limit_bytes<IsEven, NumActiveChannels>(policy) / counter_size_bytes / NumActiveChannels;
  if (num_bins <= static_smem_max_bins)
  {
    return privatization_mode::static_smem;
  }
  if (num_bins <= dynamic_smem_max_bins)
  {
    return privatization_mode::dynamic_smem;
  }
  return privatization_mode::gmem;
}

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, HistogramPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  bool sample_is_primitive; //!< Whether the sample opts into CUB's primitive-type tuning category
  // Kept separately from sample_size_bytes to preserve the serialized C Parallel selector layout.
  int sample_size;
  int counter_size_bytes;
  int sample_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;
  type_t sample_type;

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    // SM107 adds dedicated single-channel HistogramRange sweep tuning for 16-bit samples and non-floating-point
    // 32-bit samples. Other cases retain the established SM90 policy.
    if (cc >= ::cuda::compute_capability{10, 7} && cc < ::cuda::compute_capability{11, 0} && num_channels == 1
        && num_active_channels == 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
        && !is_even)
    {
      auto sweep = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      if (sample_size_bytes == 2)
      {
        sweep = HistogramPrivatizationPolicy{1024, 9, 4, BLOCK_LOAD_STRIPED, LOAD_DEFAULT, true, false};
      }
      else if (sample_size_bytes == 4 && sample_type != type_t::float32)
      {
        sweep = HistogramPrivatizationPolicy{992, 9, 2, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, true, false};
      }

      if (sample_size_bytes == 2 || (sample_size_bytes == 4 && sample_type != type_t::float32))
      {
        return HistogramPolicy{sweep, sweep, sweep, 256, 1024, 0, 0, 0, 0, 0, 0, 8192};
      }
    }

    // SM100 and SM120 use the autoresearch launch shapes. Their dynamic-SMEM budgets differ because SM100 permits
    // 227 KiB per block while SM120 permits 99 KiB per block.
    if (cc == ::cuda::compute_capability{10, 0} || cc == ::cuda::compute_capability{12, 0})
    {
      const bool is_sm120       = cc == ::cuda::compute_capability{12, 0};
      const bool single_channel = num_channels == 1 && num_active_channels == 1;
      auto gmem = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};

      // Single-channel primitive samples with 32-bit counters use their per-sample-width tuning.
      if (single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive)
      {
        // Eight-bit EVEN and RANGE histograms retain the dedicated SM100 tunings already in main.
        if (sample_size_bytes == 1)
        {
          gmem = is_even ? HistogramPrivatizationPolicy{928, 12, 4, BLOCK_LOAD_DIRECT, LOAD_CA, false, false}
                         : HistogramPrivatizationPolicy{448, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
        }
        // Sixteen-bit samples retain the SM90 tuning because autoresearch did not improve it.
        else if (sample_size_bytes == 2)
        {
          gmem = HistogramPrivatizationPolicy{960, 10, 4, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false};
        }
        // Thirty-two-bit samples use the best sweep shape measured by autoresearch.
        else if (sample_size_bytes == 4)
        {
          gmem = HistogramPrivatizationPolicy{768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
        }
        // Sixty-four-bit samples use the best sweep shape measured by autoresearch.
        else if (sample_size_bytes == 8)
        {
          gmem = HistogramPrivatizationPolicy{768, 6, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
        }
      }

      auto static_smem = gmem;
      const bool range_multi_static =
        !is_even && num_channels > 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive;
      const bool range_u32_static =
        !is_even && single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
        && sample_size_bytes == 4;
      const bool range_u64_static =
        !is_even && single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
        && sample_size_bytes == 8;
      // Multi-channel and 64-bit-sample RANGE favor narrower blocks in the static-SMEM tier.
      if (range_multi_static || range_u64_static)
      {
        static_smem.threads_per_block = 384;
      }
      // Thirty-two-bit-sample RANGE retains the wider block that won in the static-SMEM tier.
      else if (range_u32_static)
      {
        static_smem.threads_per_block = 768;
      }
      // Sixty-four-bit-sample RANGE recovers the higher static-tier items-per-thread count.
      if (range_u64_static)
      {
        static_smem.items_per_thread = t_scale(16);
      }

      // All storage thresholds are byte budgets. Dispatch derives the corresponding
      // bin limits from the local counter width and active channel count.
      constexpr int max_privatized_static_smem_bytes                    = 1024;
      constexpr int max_privatized_dynamic_smem_sm100_bytes             = 228352;
      constexpr int max_privatized_dynamic_smem_sm120_bytes             = 99 * 1024;
      constexpr int max_privatized_dynamic_smem_range_bytes_per_channel = 8192;
      constexpr int max_privatized_dynamic_smem_even_bytes_per_channel  = 32768;
      constexpr int init_threads_per_block                              = 256;
      constexpr int max_output_histogram_bytes_for_init_kernel_pdl      = 8192;
      const int max_privatized_dynamic_smem_single_channel_bytes =
        is_sm120 ? max_privatized_dynamic_smem_sm120_bytes : max_privatized_dynamic_smem_sm100_bytes;

      const bool supports_dynamic_smem =
        counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive;
      const bool has_single_channel_dynamic_smem =
        supports_dynamic_smem && single_channel
        && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8);
      const bool has_multi_channel_dynamic_smem = supports_dynamic_smem && num_channels > 1;
      const int dynamic_smem_single_channel_bytes =
        has_single_channel_dynamic_smem ? max_privatized_dynamic_smem_single_channel_bytes : 0;
      const int dynamic_smem_multi_channel_range_bytes =
        has_single_channel_dynamic_smem || (has_multi_channel_dynamic_smem && !is_even)
          ? max_privatized_dynamic_smem_range_bytes_per_channel * num_active_channels
          : 0;
      int dynamic_smem_multi_channel_even_bytes = 0;
      if (has_multi_channel_dynamic_smem && is_even)
      {
        dynamic_smem_multi_channel_even_bytes =
          max_privatized_dynamic_smem_even_bytes_per_channel * num_active_channels;
        if (is_sm120)
        {
          dynamic_smem_multi_channel_even_bytes =
            (::cuda::std::min) (dynamic_smem_multi_channel_even_bytes, max_privatized_dynamic_smem_sm120_bytes);
        }
      }
      const int init_kernel_pdl_trigger_bytes =
        single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
            && (sample_size_bytes == 1 || sample_size_bytes == 2 || sample_size_bytes == 4 || sample_size_bytes == 8)
          ? max_output_histogram_bytes_for_init_kernel_pdl
          : 0;
      auto policy = HistogramPolicy{
        gmem,
        static_smem,
        gmem,
        init_threads_per_block,
        max_privatized_static_smem_bytes,
        dynamic_smem_single_channel_bytes,
        range_multi_static || range_u64_static ? 3 : 0,
        dynamic_smem_multi_channel_range_bytes,
        has_multi_channel_dynamic_smem && is_even && num_active_channels == 2
          ? dynamic_smem_multi_channel_even_bytes
          : 0,
        has_multi_channel_dynamic_smem && is_even && num_active_channels == 3
          ? dynamic_smem_multi_channel_even_bytes
          : 0,
        has_multi_channel_dynamic_smem && is_even && num_active_channels == 4
          ? dynamic_smem_multi_channel_even_bytes
          : 0,
        init_kernel_pdl_trigger_bytes};

      const bool cooperative_single_channel =
        single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
        && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8);
      const bool cooperative_multi_channel =
        num_channels >= 2 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive;
      if (cooperative_single_channel || cooperative_multi_channel)
      {
        const bool use_full_smem_capacity = single_channel || (is_even && num_active_channels <= 3);
        const int candidate_smem_bytes =
          use_full_smem_capacity ? max_privatized_dynamic_smem_sm100_bytes
          : is_even              ? max_privatized_dynamic_smem_even_bytes_per_channel * num_active_channels
                                 : max_privatized_dynamic_smem_range_bytes_per_channel * num_active_channels;
        policy.high_bin_algorithm   = HistogramHighBinAlgorithm::cooperative;
        policy.high_bin_cache       = HistogramCacheAlgorithm::single_probe;
        policy.high_bin_spill       = HistogramSpillAlgorithm::global_memory_privatized;
        policy.high_bin_aggregation = HistogramAggregationAlgorithm::rle;
        policy.high_bin_cache_bytes_per_channel =
          single_channel
            ? 65536
            : (is_even || sample_size_bytes == 8 ? 2048 : 1024)
                * (int{sizeof(::cuda::std::uint32_t)} + 4 * counter_size_bytes);
        policy.high_bin_cache_count_replicas             = single_channel ? 1 : 4;
        policy.high_bin_cache_cuckoo_max_histogram_bytes = 1048576;
        policy.high_bin_items_per_thread                 = 4;
        const int max_privatized_dynamic_smem_bytes =
          is_sm120 ? max_privatized_dynamic_smem_sm120_bytes : max_privatized_dynamic_smem_sm100_bytes;
        policy.high_bin_min_histogram_bytes =
          (::cuda::std::min) (candidate_smem_bytes, max_privatized_dynamic_smem_bytes);
        policy.high_bin_blocks_per_sm = single_channel || (!is_even && sample_size_bytes == 4) ? 2 : 1;
        policy.high_bin_grid_items_per_block =
          single_channel && sample_size_bytes == 1 ? policy.gmem.threads_per_block * policy.gmem.items_per_thread
          : single_channel                         ? 768 * t_scale(12)
                                                   : 1024 * (is_even ? t_scale(8) : t_scale(16));
      }
      return policy;
    }

    // SM90 uses its established single-channel 8-bit and 16-bit specializations.
    if (cc >= ::cuda::compute_capability{9, 0})
    {
      auto sweep = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      // Single-channel primitive samples with 32-bit counters use the established SM90 specializations.
      if (num_channels == 1 && num_active_channels == 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)}
          && sample_is_primitive)
      {
        // Eight-bit samples use the tuned SM90 sweep.
        if (sample_size_bytes == 1)
        {
          sweep = HistogramPrivatizationPolicy{768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
        }
        // Sixteen-bit samples use the tuned SM90 sweep.
        else if (sample_size_bytes == 2)
        {
          sweep = HistogramPrivatizationPolicy{960, 10, 4, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false};
        }
      }
      constexpr int max_privatized_static_smem_bytes               = 1024;
      constexpr int init_threads_per_block                         = 256;
      constexpr int max_output_histogram_bytes_for_init_kernel_pdl = 8192;
      const int init_kernel_pdl_trigger_bytes =
        num_channels == 1 && num_active_channels == 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)}
            && sample_is_primitive && (sample_size_bytes == 1 || sample_size_bytes == 2)
          ? max_output_histogram_bytes_for_init_kernel_pdl
          : 0;
      return HistogramPolicy{
        sweep,
        sweep,
        sweep,
        init_threads_per_block,
        max_privatized_static_smem_bytes,
        0,
        0,
        0,
        0,
        0,
        0,
        init_kernel_pdl_trigger_bytes};
    }

    // Architectures before SM90 use the longstanding generic histogram tuning.
    const auto sweep = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
    return HistogramPolicy{sweep, sweep, sweep, 256, 1024, 0, 0, 0, 0, 0, 0, 0};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(histogram_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    return policy_selector{
      is_primitive_v<SampleT>,
      int{sizeof(SampleT)},
      int{sizeof(CounterT)},
      int{sizeof(SampleT)},
      NumChannels,
      NumActiveChannels,
      IsEven,
      classify_type<SampleT>}(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
