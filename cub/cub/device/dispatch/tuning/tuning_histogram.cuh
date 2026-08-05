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
inline constexpr int pre_sm100_static_smem_max_bins       = 256;
inline constexpr int sm100_static_smem_max_bins           = 512;
inline constexpr int pdl_trigger_max_bins                 = 2048;
inline constexpr int sm100_opt_in_smem_bytes              = 232448;
inline constexpr int sm100_non_histogram_smem_reserve     = 4096;
inline constexpr int sm100_dynamic_smem_max_bytes         = sm100_opt_in_smem_bytes - sm100_non_histogram_smem_reserve;
inline constexpr int sm100_range_dynamic_smem_max_bins    = 2048;
inline constexpr int sm100_even_2ch_dynamic_smem_max_bins = 28544;
inline constexpr int sm100_even_3ch_dynamic_smem_max_bins = 19029;
inline constexpr int sm100_even_4ch_dynamic_smem_max_bins = 8192;

template <class PrivatizationMode>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const HistogramKernelConfig&
kernel_config(const HistogramPolicy& policy, PrivatizationMode)
{
  if constexpr (is_privatized_static_smem_v<PrivatizationMode>)
  {
    return policy.static_smem.kernel;
  }
  else if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)
  {
    return policy.dynamic_smem.kernel;
  }
  else
  {
    static_assert(is_privatized_gmem_v<PrivatizationMode>);
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

// sample_size 2/4/8 retain the SM90 launch shape in the legacy policy hub.

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
    using AgentHistogramPolicyT = agent_histogram_policy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, false>;
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

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int init_kernel_pdl_trigger_max_bins =
      NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
          && (sizeof(SampleT) == 1 || sizeof(SampleT) == 2 || sizeof(SampleT) == 4 || sizeof(SampleT) == 8)
        ? pdl_trigger_max_bins
        : 0;
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
  int sample_size_bytes;
  int counter_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto default_kernel_config() const -> HistogramKernelConfig
  {
    return {384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm90_kernel_config() const -> HistogramKernelConfig
  {
    if (num_channels == 1 && num_active_channels == 1 && counter_size_bytes == 4 && sample_is_primitive)
    {
      if (sample_size_bytes == 1)
      {
        return {768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
      }
      if (sample_size_bytes == 2)
      {
        return {960, 10, 4, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false};
      }
    }
    return default_kernel_config();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm100_kernel_config() const -> HistogramKernelConfig
  {
    if (num_channels >= 2 && counter_size_bytes == 4 && sample_is_primitive)
    {
      return {1024, t_scale(is_even ? 8 : 16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
    }
    if (num_channels == 1 && num_active_channels == 1 && counter_size_bytes == 4 && sample_is_primitive)
    {
      if (sample_size_bytes == 1)
      {
        return is_even ? HistogramKernelConfig{928, 12, 4, BLOCK_LOAD_DIRECT, LOAD_CA, false, false}
                       : HistogramKernelConfig{448, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
      }
      if (sample_size_bytes == 4)
      {
        return {768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      }
      if (sample_size_bytes == 8)
      {
        return {768, 6, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      }
    }
    return sm90_kernel_config();
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    const bool single_channel = num_channels == 1 && num_active_channels == 1;
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      const HistogramKernelConfig kernel = sm100_kernel_config();
      const bool range_multi_static = !is_even && num_channels >= 2 && counter_size_bytes == 4 && sample_is_primitive;
      const bool range_u32_static =
        !is_even && single_channel && counter_size_bytes == 4 && sample_is_primitive && sample_size_bytes == 4;
      const bool range_u64_static =
        !is_even && single_channel && counter_size_bytes == 4 && sample_is_primitive && sample_size_bytes == 8;
      HistogramKernelConfig static_kernel = kernel;
      if (range_multi_static || range_u64_static)
      {
        static_kernel.threads_per_block = 384;
      }
      else if (range_u32_static)
      {
        static_kernel.threads_per_block = 768;
      }
      if (range_u64_static)
      {
        static_kernel.items_per_thread = t_scale(16);
      }

      const bool has_dynamic_smem_tuning =
        counter_size_bytes == 4 && sample_is_primitive
        && ((single_channel && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8))
            || num_channels >= 2);
      const int dynamic_smem_bytes    = has_dynamic_smem_tuning ? sm100_dynamic_smem_max_bytes : 0;
      const int dynamic_range_bins    = has_dynamic_smem_tuning ? sm100_range_dynamic_smem_max_bins : 0;
      const int dynamic_even_2ch_bins = has_dynamic_smem_tuning ? sm100_even_2ch_dynamic_smem_max_bins : 0;
      const int dynamic_even_3ch_bins = has_dynamic_smem_tuning ? sm100_even_3ch_dynamic_smem_max_bins : 0;
      const int dynamic_even_4ch_bins = has_dynamic_smem_tuning ? sm100_even_4ch_dynamic_smem_max_bins : 0;
      const int pdl_bins =
        single_channel && counter_size_bytes == 4 && sample_is_primitive
            && (sample_size_bytes == 1 || sample_size_bytes == 2 || sample_size_bytes == 4 || sample_size_bytes == 8)
          ? pdl_trigger_max_bins
          : 0;
      return {
        kernel,
        {static_kernel,
         sm100_static_smem_max_bins * counter_size_bytes * num_active_channels,
         range_multi_static || range_u64_static ? 3 : 0},
        {kernel,
         dynamic_smem_bytes,
         dynamic_range_bins,
         dynamic_even_2ch_bins,
         dynamic_even_3ch_bins,
         dynamic_even_4ch_bins},
        pdl_bins};
    }

    const HistogramKernelConfig kernel =
      cc >= ::cuda::compute_capability{9, 0} ? sm90_kernel_config() : default_kernel_config();
    const int pdl_bins =
      cc >= ::cuda::compute_capability{9, 0} && single_channel && counter_size_bytes == 4 && sample_is_primitive
          && (sample_size_bytes == 1 || sample_size_bytes == 2)
        ? pdl_trigger_max_bins
        : 0;
    return {kernel,
            {kernel, pre_sm100_static_smem_max_bins * counter_size_bytes * num_active_channels, 0},
            {kernel, 0, 0, 0, 0, 0},
            pdl_bins};
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
      is_primitive_v<SampleT>, int{sizeof(SampleT)}, int{sizeof(CounterT)}, NumChannels, NumActiveChannels, IsEven}(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
