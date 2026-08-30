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
_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, HistogramHighBinAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}

_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, HistogramCacheAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}

_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, HistogramSpillAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}

_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, HistogramAggregationAlgorithm value)
{
  return os << detail::histogram::to_string(value);
}
#endif // _CCCL_HOSTED()

//! The tuning policy for all algorithms in @ref DeviceHistogram.
struct HistogramPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int pixels_per_thread; //!< Number of pixels processed per thread
  int vec_size; //!< Vectorization size for loading samples
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading samples from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading samples from global memory
  bool rle_compress; //!< Whether to perform localized RLE to compress samples before histogramming
  BlockHistogramMemoryPreference mem_preference; //!< Whether to prefer privatized shared-memory or global-memory bins,
                                                 //!< or a mix of both
  bool use_work_stealing; //!< Whether to dequeue tiles from a global work queue
  int init_kernel_pdl_trigger_max_bins; //!< Maximum number of bins for the init kernel to trigger the histogram kernel
                                        //!< early using PDL
  HistogramHighBinAlgorithm high_bin_algorithm       = HistogramHighBinAlgorithm::cooperative;
  HistogramCacheAlgorithm high_bin_cache             = HistogramCacheAlgorithm::cuckoo;
  HistogramSpillAlgorithm high_bin_spill             = HistogramSpillAlgorithm::output;
  HistogramAggregationAlgorithm high_bin_aggregation = HistogramAggregationAlgorithm::warp_coalesced;
  int high_bin_cache_entries_per_channel             = 2048;
  int high_bin_cache_count_replicas                  = 1;
  int high_bin_cache_cuckoo_max_bins                 = 262144;
  int high_bin_pixels_per_thread                     = 4;

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.pixels_per_thread == rhs.pixels_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.mem_preference == rhs.mem_preference && lhs.use_work_stealing == rhs.use_work_stealing
        && lhs.init_kernel_pdl_trigger_max_bins == rhs.init_kernel_pdl_trigger_max_bins
        && lhs.high_bin_algorithm == rhs.high_bin_algorithm && lhs.high_bin_cache == rhs.high_bin_cache
        && lhs.high_bin_spill == rhs.high_bin_spill && lhs.high_bin_aggregation == rhs.high_bin_aggregation
        && lhs.high_bin_cache_entries_per_channel == rhs.high_bin_cache_entries_per_channel
        && lhs.high_bin_cache_count_replicas == rhs.high_bin_cache_count_replicas
        && lhs.high_bin_cache_cuckoo_max_bins == rhs.high_bin_cache_cuckoo_max_bins
        && lhs.high_bin_pixels_per_thread == rhs.high_bin_pixels_per_thread;
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
        << "HistogramPolicy { .threads_per_block = " << p.threads_per_block << ", .pixels_per_thread = "
        << p.pixels_per_thread << ", .vec_size = " << p.vec_size << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rle_compress = " << p.rle_compress
        << ", .mem_preference = " << p.mem_preference << ", .use_work_stealing = " << p.use_work_stealing
        << ", .init_kernel_pdl_trigger_max_bins = " << p.init_kernel_pdl_trigger_max_bins
        << ", .high_bin_algorithm = " << p.high_bin_algorithm << ", .high_bin_cache = " << p.high_bin_cache
        << ", .high_bin_spill = " << p.high_bin_spill << ", .high_bin_aggregation = " << p.high_bin_aggregation
        << ", .high_bin_cache_entries_per_channel = " << p.high_bin_cache_entries_per_channel
        << ", .high_bin_cache_count_replicas = " << p.high_bin_cache_count_replicas
        << ", .high_bin_cache_cuckoo_max_bins = " << p.high_bin_cache_cuckoo_max_bins
        << ", .high_bin_pixels_per_thread = " << p.high_bin_pixels_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::histogram
{
// TODO(bgruber): drop in CCCL 4.0
enum class primitive_sample
{
  no,
  yes
};

// TODO(bgruber): drop in CCCL 4.0
enum class sample_size
{
  _1,
  _2,
  _4,
  _8,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
enum class counter_size
{
  _4,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
template <class T>
_CCCL_HOST_DEVICE_API constexpr primitive_sample is_primitive_sample()
{
  return is_primitive<T>::value ? primitive_sample::yes : primitive_sample::no;
}

// TODO(bgruber): drop in CCCL 4.0
template <class CounterT>
_CCCL_HOST_DEVICE_API constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT>
_CCCL_HOST_DEVICE_API constexpr sample_size classify_sample_size()
{
  return sizeof(SampleT) == 1 ? sample_size::_1 : sizeof(SampleT) == 2 ? sample_size::_2 : sample_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm90_tuning;

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  static constexpr int threads = 768;
  static constexpr int items   = 12;

  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress      = false;
  static constexpr bool use_work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads = 960;
  static constexpr int items   = 10;

  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress      = true;
  static constexpr bool use_work_stealing = false;
};

// TODO(bgruber): drop in CCCL 4.0
template <bool IsEven,
          class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm100_tuning;

// even
template <class SampleT>
struct sm100_tuning<true, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 928;
  static constexpr bool rle_compress                             = false;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_CA;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks

// range
template <class SampleT>
struct sm100_tuning<false, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 448;
  static constexpr bool rle_compress                             = false;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks

// multi.even and multi.range: none of the found tunings surpassed the SM90 tuning during verification benchmarks

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  // TODO(bgruber): move inside t_scale in C++14
  static constexpr int v_scale = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);

  _CCCL_HOST_DEVICE_API static constexpr int t_scale(int nominalItemsPerThread)
  {
    return (::cuda::std::max) (nominalItemsPerThread / NumActiveChannels / v_scale, 1);
  }

  // SM50
  struct Policy500 : detail::chained_policy<500, Policy500, Policy500>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      agent_histogram_policy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
  };

  // SM90
  struct Policy900 : detail::chained_policy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> agent_histogram_policy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                Tuning::rle_compress,
                                Tuning::mem_preference,
                                Tuning::use_work_stealing>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    static constexpr int init_kernel_pdl_trigger_max_bins = 2048;
  };

  struct Policy1000 : detail::chained_policy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int) -> agent_histogram_policy<
      Tuning::threads,
      Tuning::items,
      Tuning::load_algorithm,
      Tuning::load_modifier,
      Tuning::rle_compress,
      Tuning::mem_preference,
      Tuning::use_work_stealing,
      Tuning::vec_size>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int init_kernel_pdl_trigger_max_bins = 2048;
  };

  using MaxPolicy = Policy1000;
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, HistogramPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  bool sample_is_primitive;
  int sample_size;
  int counter_size;
  int sample_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 1)
      {
        if (is_even)
        {
          // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
          return HistogramPolicy{
            928,
            12,
            1 << 2,
            BLOCK_LOAD_DIRECT,
            LOAD_CA,
            false,
            SMEM,
            false,
            2048,
            HistogramHighBinAlgorithm::cooperative,
            HistogramCacheAlgorithm::cuckoo,
            HistogramSpillAlgorithm::output,
            HistogramAggregationAlgorithm::warp_coalesced,
            4096,
            1,
            262144,
            4};
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return HistogramPolicy{
            448,
            12,
            1 << 2,
            BLOCK_LOAD_DIRECT,
            LOAD_LDG,
            false,
            SMEM,
            false,
            2048,
            HistogramHighBinAlgorithm::cooperative,
            HistogramCacheAlgorithm::cuckoo,
            HistogramSpillAlgorithm::output,
            HistogramAggregationAlgorithm::warp_coalesced,
            2048,
            2,
            262144,
            4};
        }
      }

      // sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks
      // multi.even and multi.range: none of the found tunings surpassed the SM90 tuning during verification benchmarks
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return HistogramPolicy{
            768,
            12,
            1 << 2,
            BLOCK_LOAD_DIRECT,
            LOAD_LDG,
            false,
            SMEM,
            false,
            2048,
            HistogramHighBinAlgorithm::cooperative,
            HistogramCacheAlgorithm::cuckoo,
            HistogramSpillAlgorithm::output,
            HistogramAggregationAlgorithm::warp_coalesced,
            is_even ? 4096 : 2048,
            is_even ? 1 : 2,
            262144,
            4};
        }
        else if (sample_size == 2)
        {
          return HistogramPolicy{
            960,
            10,
            1 << 2,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            true,
            SMEM,
            false,
            2048,
            HistogramHighBinAlgorithm::cooperative,
            HistogramCacheAlgorithm::cuckoo,
            HistogramSpillAlgorithm::output,
            HistogramAggregationAlgorithm::warp_coalesced,
            is_even ? 4096 : 2048,
            is_even ? 1 : 2,
            262144,
            4};
        }
      }
    }

    // fallback from SM50
    return HistogramPolicy{
      384,
      t_scale(16),
      4,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      true,
      SMEM,
      false,
      0,
      HistogramHighBinAlgorithm::cooperative,
      HistogramCacheAlgorithm::cuckoo,
      HistogramSpillAlgorithm::output,
      HistogramAggregationAlgorithm::warp_coalesced,
      num_active_channels == 1 ? (is_even ? 4096 : 2048) : 512,
      is_even ? 1 : (num_active_channels == 1 ? 2 : 4),
      262144,
      num_active_channels == 1 ? 4 : 1};
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
    constexpr auto policies = policy_selector{
      is_primitive_v<SampleT>,
      int{sizeof(SampleT)},
      int{sizeof(CounterT)},
      int{sizeof(SampleT)},
      NumChannels,
      NumActiveChannels,
      IsEven};
    return policies(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
