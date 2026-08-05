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

//! Runtime launch configuration for one DeviceHistogram kernel.
struct HistogramKernelConfig
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  int vec_size; //!< Vectorization size for loading samples
  BlockLoadAlgorithm load_algorithm; //!< Algorithm used for loading samples
  CacheLoadModifier load_modifier; //!< Cache modifier used for loading samples
  bool rle_compress; //!< Whether to locally run-length encode samples
  bool work_stealing; //!< Whether blocks dequeue tiles from a global queue

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramKernelConfig& lhs, const HistogramKernelConfig& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.work_stealing == rhs.work_stealing;
  }
};

//! Tuning policy for the compile-time-sized shared-memory histogram kernel.
struct HistogramStaticSmemPolicy
{
  HistogramKernelConfig kernel;
  int max_privatized_smem_bytes; //!< Maximum compile-time-sized shared-memory allocation
  int min_blocks_per_sm; //!< Minimum blocks per SM requested through launch bounds

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramStaticSmemPolicy& lhs, const HistogramStaticSmemPolicy& rhs) noexcept
  {
    return lhs.kernel == rhs.kernel && lhs.max_privatized_smem_bytes == rhs.max_privatized_smem_bytes
        && lhs.min_blocks_per_sm == rhs.min_blocks_per_sm;
  }
};

//! Tuning policy for the runtime-sized shared-memory histogram kernel.
struct HistogramDynamicSmemPolicy
{
  HistogramKernelConfig kernel;
  int max_privatized_smem_bytes; //!< Maximum runtime-sized shared-memory allocation
  int range_max_bins; //!< Maximum bins per channel for multi-channel HistogramRange
  int even_2ch_max_bins; //!< Maximum bins per channel for two-channel HistogramEven
  int even_3ch_max_bins; //!< Maximum bins per channel for three-channel HistogramEven
  int even_4ch_max_bins; //!< Maximum bins per channel for four-channel HistogramEven

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramDynamicSmemPolicy& lhs, const HistogramDynamicSmemPolicy& rhs) noexcept
  {
    return lhs.kernel == rhs.kernel && lhs.max_privatized_smem_bytes == rhs.max_privatized_smem_bytes
        && lhs.range_max_bins == rhs.range_max_bins && lhs.even_2ch_max_bins == rhs.even_2ch_max_bins
        && lhs.even_3ch_max_bins == rhs.even_3ch_max_bins && lhs.even_4ch_max_bins == rhs.even_4ch_max_bins;
  }
};

//! The tuning policy for all DeviceHistogram kernel variants.
struct HistogramPolicy
{
  HistogramKernelConfig gmem;
  HistogramStaticSmemPolicy static_smem;
  HistogramDynamicSmemPolicy dynamic_smem;
  int init_kernel_pdl_trigger_max_bins; //!< Common init-kernel PDL threshold, independent of accumulation tier

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.gmem == rhs.gmem && lhs.static_smem == rhs.static_smem && lhs.dynamic_smem == rhs.dynamic_smem
        && lhs.init_kernel_pdl_trigger_max_bins == rhs.init_kernel_pdl_trigger_max_bins;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const HistogramPolicy& p)
  {
    const auto print_kernel = [&](const HistogramKernelConfig& kernel) -> ::std::ostream& {
      return os
          << "{ .threads_per_block = " << kernel.threads_per_block
          << ", .items_per_thread = " << kernel.items_per_thread << ", .vec_size = " << kernel.vec_size
          << ", .load_algorithm = " << kernel.load_algorithm << ", .load_modifier = " << kernel.load_modifier
          << ", .rle_compress = " << kernel.rle_compress << ", .work_stealing = " << kernel.work_stealing << " }";
    };
    os << "HistogramPolicy { .gmem = ";
    print_kernel(p.gmem);
    os << ", .static_smem = { .kernel = ";
    print_kernel(p.static_smem.kernel);
    os << ", .max_privatized_smem_bytes = " << p.static_smem.max_privatized_smem_bytes
       << ", .min_blocks_per_sm = " << p.static_smem.min_blocks_per_sm << " }, .dynamic_smem = { .kernel = ";
    print_kernel(p.dynamic_smem.kernel);
    return os
        << ", .max_privatized_smem_bytes = " << p.dynamic_smem.max_privatized_smem_bytes << ", .range_max_bins = "
        << p.dynamic_smem.range_max_bins << ", .even_2ch_max_bins = " << p.dynamic_smem.even_2ch_max_bins
        << ", .even_3ch_max_bins = " << p.dynamic_smem.even_3ch_max_bins
        << ", .even_4ch_max_bins = " << p.dynamic_smem.even_4ch_max_bins
        << " }, .init_kernel_pdl_trigger_max_bins = " << p.init_kernel_pdl_trigger_max_bins << " }";
  }
#endif
};

namespace detail::histogram
{
// Compile-time policy wrappers carried by each node in CUB's architecture
// policy chain. They pair an AgentHistogram launch policy with the storage
// limits needed by that kernel variant. `make_kernel_config` below converts
// these type-level values into the runtime `HistogramPolicy` selected by
// dispatch.
template <class AgentPolicy, int MaxPrivatizedSmemBytes, int MinBlocksPerSm = 0>
struct static_smem_chained_policy
{
  using AgentPolicyT                             = AgentPolicy;
  static constexpr int MAX_PRIVATIZED_SMEM_BYTES = MaxPrivatizedSmemBytes;
  static constexpr int MIN_BLOCKS_PER_SM         = MinBlocksPerSm;
};

template <class AgentPolicy,
          int MaxPrivatizedSmemBytes,
          int RangeMaxBins,
          int Even2chMaxBins,
          int Even3chMaxBins,
          int Even4chMaxBins>
struct dynamic_smem_chained_policy
{
  using AgentPolicyT                             = AgentPolicy;
  static constexpr int MAX_PRIVATIZED_SMEM_BYTES = MaxPrivatizedSmemBytes;
  static constexpr int RANGE_MAX_BINS            = RangeMaxBins;
  static constexpr int EVEN_2CH_MAX_BINS         = Even2chMaxBins;
  static constexpr int EVEN_3CH_MAX_BINS         = Even3chMaxBins;
  static constexpr int EVEN_4CH_MAX_BINS         = Even4chMaxBins;
};

enum class privatization_tier
{
  gmem,
  static_smem,
  dynamic_smem
};

template <privatization_tier Tier>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const HistogramKernelConfig& kernel_config(const HistogramPolicy& policy)
{
  if constexpr (Tier == privatization_tier::static_smem)
  {
    return policy.static_smem.kernel;
  }
  else if constexpr (Tier == privatization_tier::dynamic_smem)
  {
    return policy.dynamic_smem.kernel;
  }
  else
  {
    return policy.gmem;
  }
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_privatized_smem_bins(int max_privatized_smem_bytes, int counter_size, int num_active_channels)
{
  if (max_privatized_smem_bytes <= 0 || counter_size <= 0 || num_active_channels <= 0)
  {
    return 0;
  }
  return max_privatized_smem_bytes / counter_size / num_active_channels;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_privatized_static_smem_bins(const HistogramPolicy& policy, int counter_size, int num_active_channels)
{
  return max_privatized_smem_bins(policy.static_smem.max_privatized_smem_bytes, counter_size, num_active_channels);
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_privatized_dynamic_smem_bins(const HistogramPolicy& policy, int counter_size, int num_active_channels)
{
  return max_privatized_smem_bins(policy.dynamic_smem.max_privatized_smem_bytes, counter_size, num_active_channels);
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool
should_use_static_smem(const HistogramPolicy& policy, int num_bins, int counter_size, int num_active_channels)
{
  return num_bins > 0 && num_bins <= max_privatized_static_smem_bins(policy, counter_size, num_active_channels);
}

template <bool IsEven>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool
should_use_dynamic_smem(const HistogramPolicy& policy, int num_bins, int counter_size, int num_active_channels)
{
  // Single-channel limits are intentionally byte-derived: the B200 tuning
  // characterized that path through the full opt-in shared-memory budget.
  // Multi-channel paths use explicit per-channel caps in addition to the byte
  // budget because their channel-interleaved launch shapes have distinct
  // measured crossover points.
  if (num_bins <= 0)
  {
    return false;
  }

  const bool prefer_dynamic_smem = counter_size > int{sizeof(unsigned int)}
                                || !should_use_static_smem(policy, num_bins, counter_size, num_active_channels);

  int max_bins = max_privatized_dynamic_smem_bins(policy, counter_size, num_active_channels);
  if (num_active_channels > 1)
  {
    if constexpr (IsEven)
    {
      max_bins = num_active_channels == 2 ? policy.dynamic_smem.even_2ch_max_bins
               : num_active_channels == 3
                 ? policy.dynamic_smem.even_3ch_max_bins
                 : policy.dynamic_smem.even_4ch_max_bins;
    }
    else
    {
      max_bins = policy.dynamic_smem.range_max_bins;
    }
  }

  return prefer_dynamic_smem && max_bins > 0 && num_bins <= max_bins;
}

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
  return sizeof(SampleT) == 1 ? sample_size::_1
       : sizeof(SampleT) == 2 ? sample_size::_2
       : sizeof(SampleT) == 4 ? sample_size::_4
       : sizeof(SampleT) == 8
         ? sample_size::_8
         : sample_size::unknown;
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
  static constexpr int threads_per_block = 768;
  static constexpr int items_per_thread  = 12;

  static constexpr CacheLoadModifier load_modifier = LOAD_LDG;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = false;
  static constexpr bool work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads_per_block = 960;
  static constexpr int items_per_thread  = 10;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = true;
  static constexpr bool work_stealing = false;
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
  static constexpr int items_per_thread              = 12;
  static constexpr int threads_per_block             = 928;
  static constexpr bool rle_compress                 = false;
  static constexpr bool work_stealing                = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                      = 1 << 2;
};

// range
template <class SampleT>
struct sm100_tuning<false, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
  static constexpr int items_per_thread              = 12;
  static constexpr int threads_per_block             = 448;
  static constexpr bool rle_compress                 = false;
  static constexpr bool work_stealing                = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                      = 1 << 2;
};

template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{
  static constexpr int items_per_thread              = 12;
  static constexpr int threads_per_block             = 768;
  static constexpr bool rle_compress                 = true;
  static constexpr bool work_stealing                = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                      = 1 << 2;
};

template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{
  static constexpr int items_per_thread              = 6;
  static constexpr int threads_per_block             = 768;
  static constexpr bool rle_compress                 = true;
  static constexpr bool work_stealing                = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                      = 1 << 2;
};

// sample_size 2 retains the SM90 launch shape while using the SM100 shared-memory policy.

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  static constexpr int pre_sm100_static_smem_max_bins    = 256;
  static constexpr int sm100_static_smem_max_bins        = 512;
  static constexpr int pdl_trigger_max_bins              = 2048;
  static constexpr int sm100_opt_in_smem_bytes           = 232448;
  static constexpr int sm100_non_histogram_smem_reserve  = 4096;
  static constexpr int sm100_dynamic_smem_max_bytes      = sm100_opt_in_smem_bytes - sm100_non_histogram_smem_reserve;
  static constexpr int sm100_range_dynamic_smem_max_bins = 2048;
  static constexpr int sm100_even_2ch_dynamic_smem_max_bins = 28544;
  static constexpr int sm100_even_3ch_dynamic_smem_max_bins = 19029;
  static constexpr int sm100_even_4ch_dynamic_smem_max_bins = 8192;

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
    using AgentHistogramPolicyT = agent_histogram_policy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, false>;
    using GmemPolicy            = AgentHistogramPolicyT;
    using StaticSmemPolicy =
      static_smem_chained_policy<AgentHistogramPolicyT,
                                 pre_sm100_static_smem_max_bins * sizeof(CounterT) * NumActiveChannels>;
    using DynamicSmemPolicy = dynamic_smem_chained_policy<AgentHistogramPolicyT, 0, 0, 0, 0, 0>;

    static constexpr int init_kernel_pdl_trigger_max_bins = 0;
  };

  // SM90
  struct Policy900 : detail::chained_policy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> agent_histogram_policy<Tuning::threads_per_block,
                                Tuning::items_per_thread,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                Tuning::rle_compress,
                                Tuning::work_stealing>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    using GmemPolicy = AgentHistogramPolicyT;
    using StaticSmemPolicy =
      static_smem_chained_policy<AgentHistogramPolicyT,
                                 pre_sm100_static_smem_max_bins * sizeof(CounterT) * NumActiveChannels>;
    using DynamicSmemPolicy = dynamic_smem_chained_policy<AgentHistogramPolicyT, 0, 0, 0, 0, 0>;

    static constexpr int init_kernel_pdl_trigger_max_bins =
      NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
          && (sizeof(SampleT) == 1 || sizeof(SampleT) == 2)
        ? pdl_trigger_max_bins
        : 0;
  };

  struct Policy1000 : detail::chained_policy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> agent_histogram_policy<Tuning::threads_per_block,
                                Tuning::items_per_thread,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                Tuning::rle_compress,
                                Tuning::work_stealing,
                                Tuning::vec_size>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using SelectedAgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    using MultiChannelAgentHistogramPolicyT =
      agent_histogram_policy<1024, t_scale(IsEven ? 8 : 16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 4>;

    static constexpr bool use_sm100_multi_channel_policy =
      NumChannels >= 2 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value;

    using AgentHistogramPolicyT =
      ::cuda::std::_If<use_sm100_multi_channel_policy, MultiChannelAgentHistogramPolicyT, SelectedAgentHistogramPolicyT>;

    static constexpr bool has_dynamic_smem_tuning =
      sizeof(CounterT) == 4 && is_primitive<SampleT>::value
      && ((NumChannels == 1 && NumActiveChannels == 1
           && (sizeof(SampleT) == 1 || sizeof(SampleT) == 4 || sizeof(SampleT) == 8))
          || NumChannels >= 2);

    static constexpr int init_kernel_pdl_trigger_max_bins =
      NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
          && (sizeof(SampleT) == 1 || sizeof(SampleT) == 2 || sizeof(SampleT) == 4 || sizeof(SampleT) == 8)
        ? pdl_trigger_max_bins
        : 0;
    static constexpr bool use_range_multi_static_smem_policy =
      !IsEven && NumChannels >= 2 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value;
    static constexpr bool use_range_u32_static_smem_policy =
      !IsEven && NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
      && sizeof(SampleT) == 4;
    static constexpr bool use_range_u64_static_smem_policy =
      !IsEven && NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
      && sizeof(SampleT) == 8;

    static constexpr int static_smem_threads_per_block =
      use_range_multi_static_smem_policy || use_range_u64_static_smem_policy ? 384
      : use_range_u32_static_smem_policy
        ? 768
        : AgentHistogramPolicyT::BLOCK_THREADS;
    static constexpr int static_smem_items_per_thread =
      use_range_u64_static_smem_policy ? t_scale(16) : AgentHistogramPolicyT::PIXELS_PER_THREAD;
    static constexpr int static_smem_min_blocks_per_sm =
      use_range_multi_static_smem_policy || use_range_u64_static_smem_policy ? 3 : 0;

    using StaticSmemAgentPolicyT = agent_histogram_policy<
      static_smem_threads_per_block,
      static_smem_items_per_thread,
      AgentHistogramPolicyT::LOAD_ALGORITHM,
      AgentHistogramPolicyT::LOAD_MODIFIER,
      AgentHistogramPolicyT::IS_RLE_COMPRESS,
      AgentHistogramPolicyT::IS_WORK_STEALING,
      AgentHistogramPolicyT::VEC_SIZE,
      sm100_static_smem_max_bins * sizeof(CounterT) * NumActiveChannels>;

    using GmemPolicy = AgentHistogramPolicyT;
    using StaticSmemPolicy =
      static_smem_chained_policy<StaticSmemAgentPolicyT,
                                 sm100_static_smem_max_bins * sizeof(CounterT) * NumActiveChannels,
                                 static_smem_min_blocks_per_sm>;

    static constexpr int dynamic_smem_max_bytes = has_dynamic_smem_tuning ? sm100_dynamic_smem_max_bytes : 0;
    using DynamicSmemPolicy =
      dynamic_smem_chained_policy<AgentHistogramPolicyT,
                                  dynamic_smem_max_bytes,
                                  has_dynamic_smem_tuning ? sm100_range_dynamic_smem_max_bins : 0,
                                  has_dynamic_smem_tuning ? sm100_even_2ch_dynamic_smem_max_bins : 0,
                                  has_dynamic_smem_tuning ? sm100_even_3ch_dynamic_smem_max_bins : 0,
                                  has_dynamic_smem_tuning ? sm100_even_4ch_dynamic_smem_max_bins : 0>;
  };

  using MaxPolicy = Policy1000;
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, HistogramPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

template <class AgentPolicy>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_kernel_config() -> HistogramKernelConfig
{
  return {AgentPolicy::BLOCK_THREADS,
          AgentPolicy::PIXELS_PER_THREAD,
          AgentPolicy::VEC_SIZE,
          AgentPolicy::LOAD_ALGORITHM,
          AgentPolicy::LOAD_MODIFIER,
          AgentPolicy::IS_RLE_COMPRESS,
          AgentPolicy::IS_WORK_STEALING};
}

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm100_policy(HistogramPolicy policy) const -> HistogramPolicy
  {
    policy.dynamic_smem_bytes             = sm100_dynamic_smem_bytes;
    policy.dynamic_smem_range_max_bins    = sm100_dynamic_smem_range_max_bins;
    policy.dynamic_smem_even_2ch_max_bins = sm100_dynamic_smem_even_2ch_max_bins;
    policy.dynamic_smem_even_3ch_max_bins = sm100_dynamic_smem_even_3ch_max_bins;
    policy.dynamic_smem_even_4ch_max_bins = sm100_dynamic_smem_even_4ch_max_bins;
    return policy;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool has_sm100_dynamic_smem_tuning() const
  {
    return sample_is_primitive && counter_size == 4
        && ((num_channels == 1 && num_active_channels == 1
             && (sample_size == 1 || sample_size == 4 || sample_size == 8))
            || num_channels >= 2);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    if (cc >= ::cuda::compute_capability{10, 7} && cc < ::cuda::compute_capability{11, 0})
    {
      if (num_channels == 4 && num_active_channels == 3 && counter_size == 4 && sample_is_primitive && is_even)
      {
        if (sample_size == 1)
        {
          // ipt_7.tpb_128.rle_0.ws_0.mem_1.ld_1.laid_2.vec_0 1.005  0.991  1.537  2.076
          return HistogramPolicy{128, 7, 1, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 2048};
        }
        if (sample_size == 2)
        {
          // ipt_7.tpb_256.rle_1.ws_0.mem_1.ld_0.laid_0.vec_0 0.937  1.014  1.145  1.126
          return HistogramPolicy{256, 7, 1, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 2048};
        }
      }
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && !is_even)
      {
        if (sample_size == 2)
        {
          // ipt_9.tpb_1024.rle_1.ws_0.mem_1.ld_0.laid_2.vec_2 1.035  1.036  1.064  1.051
          return HistogramPolicy{1024, 9, 1 << 2, BLOCK_LOAD_STRIPED, LOAD_DEFAULT, true, SMEM, false, 2048};
        }
        if (sample_size == 4 && sample_type != type_t::float32)
        {
          // ipt_9.tpb_992.rle_1.ws_0.mem_1.ld_0.laid_1.vec_1 1.684  1.426  1.392  1.170
          return HistogramPolicy{992, 9, 1 << 1, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, true, SMEM, false, 2048};
        }
        // float32, 8-byte and 16-byte samples: no clean sm107 candidate, fall through
      }
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && is_even)
      {
        // 1-byte samples: every searched candidate costs ~25% at 2^16/2^20, intentionally left untuned
        // 2-byte samples: every searched candidate costs ~20% at 2^16/2^20 with 2048 bins, intentionally left untuned
        // 4-byte samples: every searched candidate regresses the 2048-bin class, intentionally left untuned
        if (sample_size == 8)
        {
          if (sample_type == type_t::float64)
          {
            // ipt_16.tpb_512.rle_1.ws_0.mem_1.ld_0.laid_2.vec_0 0.974  0.995  1.138  1.181
            return HistogramPolicy{512, 16, 1, BLOCK_LOAD_STRIPED, LOAD_DEFAULT, true, SMEM, false, 2048};
          }
          // ipt_11.tpb_512.rle_1.ws_0.mem_1.ld_2.laid_2.vec_0 0.943  1.014  1.135  1.199
          return HistogramPolicy{512, 11, 1, BLOCK_LOAD_STRIPED, LOAD_CA, true, SMEM, false, 2048};
        }
      }
    }

    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 1)
      {
        if (is_even)
        {
          // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
          return sm100_policy(HistogramPolicy{928, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_CA, false, false, 2048});
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return sm100_policy(HistogramPolicy{448, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false, 2048});
        }
      }

      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive
          && (sample_size == 4 || sample_size == 8))
      {
        if (is_even)
        {
          return sm100_policy(
            HistogramPolicy{768, t_scale(12), 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 2048});
        }

        const int static_threads    = sample_size == 8 ? 384 : 768;
        const int static_items      = sample_size == 8 ? t_scale(16) : 0;
        const int static_min_blocks = static_threads < 512 ? 3 : 0;
        return sm100_policy(HistogramPolicy{
          768,
          t_scale(12),
          1 << 2,
          BLOCK_LOAD_DIRECT,
          LOAD_LDG,
          true,
          false,
          2048,
          0,
          static_threads,
          static_items,
          static_min_blocks});
      }

      if (num_channels >= 2 && counter_size == 4 && sample_is_primitive)
      {
        if (is_even)
        {
          return sm100_policy(HistogramPolicy{1024, t_scale(8), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 0});
        }
        return sm100_policy(
          HistogramPolicy{1024, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 0, 0, 384, 0, 3});
      }

      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 2)
      {
        return HistogramPolicy{960, 10, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false, 2048};
      }

      auto fallback = HistogramPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 0};
      return has_sm100_dynamic_smem_tuning() ? sm100_policy(fallback) : fallback;
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return HistogramPolicy{768, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false, 2048};
        }
        else if (sample_size == 2)
        {
          return HistogramPolicy{960, 10, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false, 2048};
        }
      }
    }

    // fallback from SM50
    return HistogramPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false, 0};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(histogram_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class ActivePolicy>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto convert_chained_policy() -> HistogramPolicy
{
  using static_smem  = typename ActivePolicy::StaticSmemPolicy;
  using dynamic_smem = typename ActivePolicy::DynamicSmemPolicy;
  return {
    make_kernel_config<typename ActivePolicy::GmemPolicy>(),
    {make_kernel_config<typename static_smem::AgentPolicyT>(),
     static_smem::MAX_PRIVATIZED_SMEM_BYTES,
     static_smem::MIN_BLOCKS_PER_SM},
    {make_kernel_config<typename dynamic_smem::AgentPolicyT>(),
     dynamic_smem::MAX_PRIVATIZED_SMEM_BYTES,
     dynamic_smem::RANGE_MAX_BINS,
     dynamic_smem::EVEN_2CH_MAX_BINS,
     dynamic_smem::EVEN_3CH_MAX_BINS,
     dynamic_smem::EVEN_4CH_MAX_BINS},
    ActivePolicy::init_kernel_pdl_trigger_max_bins};
}

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    using hub = policy_hub<SampleT, CounterT, NumChannels, NumActiveChannels, IsEven>;
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      return convert_chained_policy<typename hub::Policy1000>();
    }
    if (cc >= ::cuda::compute_capability{9, 0})
    {
      return convert_chained_policy<typename hub::Policy900>();
    }
    return convert_chained_policy<typename hub::Policy500>();
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
