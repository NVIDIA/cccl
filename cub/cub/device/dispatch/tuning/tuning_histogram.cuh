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

//! The tuning policy for all DeviceHistogram kernel variants.
struct HistogramPolicy
{
  struct Kernel
  {
    int threads_per_block; //!< Number of threads in a CUDA block
    int items_per_thread; //!< Number of items processed per thread
    int vec_size; //!< Vectorization size for loading samples
    BlockLoadAlgorithm load_algorithm; //!< Algorithm used for loading samples
    CacheLoadModifier load_modifier; //!< Cache modifier used for loading samples
    bool rle_compress; //!< Whether to locally run-length encode samples
    bool work_stealing; //!< Whether blocks dequeue tiles from a global queue

    [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const Kernel& lhs, const Kernel& rhs) noexcept
    {
      return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
          && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
          && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
          && lhs.work_stealing == rhs.work_stealing;
    }
  };

  struct StaticSmem
  {
    Kernel kernel;
    int max_privatized_smem_bytes; //!< Maximum compile-time-sized shared-memory allocation
    int min_blocks_per_sm; //!< Minimum blocks per SM requested through launch bounds

    [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
    operator==(const StaticSmem& lhs, const StaticSmem& rhs) noexcept
    {
      return lhs.kernel == rhs.kernel && lhs.max_privatized_smem_bytes == rhs.max_privatized_smem_bytes
          && lhs.min_blocks_per_sm == rhs.min_blocks_per_sm;
    }
  };

  struct DynamicSmem
  {
    Kernel kernel;
    int max_privatized_smem_bytes; //!< Maximum runtime-sized shared-memory allocation
    int range_max_bins; //!< Maximum bins per channel for multi-channel HistogramRange
    int even_2ch_max_bins; //!< Maximum bins per channel for two-channel HistogramEven
    int even_3ch_max_bins; //!< Maximum bins per channel for three-channel HistogramEven
    int even_4ch_max_bins; //!< Maximum bins per channel for four-channel HistogramEven

    [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
    operator==(const DynamicSmem& lhs, const DynamicSmem& rhs) noexcept
    {
      return lhs.kernel == rhs.kernel && lhs.max_privatized_smem_bytes == rhs.max_privatized_smem_bytes
          && lhs.range_max_bins == rhs.range_max_bins && lhs.even_2ch_max_bins == rhs.even_2ch_max_bins
          && lhs.even_3ch_max_bins == rhs.even_3ch_max_bins && lhs.even_4ch_max_bins == rhs.even_4ch_max_bins;
    }
  };

  Kernel gmem;
  StaticSmem static_smem;
  DynamicSmem dynamic_smem;
  int init_kernel_pdl_trigger_max_bins; //!< Common init-kernel PDL threshold, independent of accumulation tier

  template <class PrivatizationMode>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const Kernel& kernel(PrivatizationMode) const
  {
    if constexpr (detail::histogram::is_privatized_static_smem_v<PrivatizationMode>)
    {
      return static_smem.kernel;
    }
    else if constexpr (detail::histogram::is_privatized_dynamic_smem_v<PrivatizationMode>)
    {
      return dynamic_smem.kernel;
    }
    else
    {
      static_assert(detail::histogram::is_privatized_gmem_v<PrivatizationMode>);
      return gmem;
    }
  }

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
    const auto print_kernel = [&](const Kernel& kernel) -> ::std::ostream& {
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
// All DeviceHistogram tuning values live in this block. Keep architecture
// selection and policy construction below free of unexplained numeric values.
inline constexpr auto sm90                                = ::cuda::compute_capability{9, 0};
inline constexpr auto sm100                               = ::cuda::compute_capability{10, 0};
inline constexpr int supported_counter_bytes              = 4;
inline constexpr int sample_u8_bytes                      = 1;
inline constexpr int sample_u16_bytes                     = 2;
inline constexpr int sample_u32_bytes                     = 4;
inline constexpr int sample_u64_bytes                     = 8;
inline constexpr int single_channel_count                 = 1;
inline constexpr int first_multi_channel_count            = 2;
inline constexpr int two_active_channels                  = 2;
inline constexpr int three_active_channels                = 3;
inline constexpr int four_active_channels                 = 4;
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
inline constexpr int legacy_privatized_smem_bins          = 256;
inline constexpr int byte_sample_privatized_levels        = legacy_privatized_smem_bins + 1;
inline constexpr int init_threads_per_block               = 256;
inline constexpr int default_threads_per_block            = 384;
inline constexpr int default_nominal_items_per_thread     = 16;
inline constexpr int default_vec_size                     = 4;
inline constexpr int sm90_u8_threads_per_block            = 768;
inline constexpr int sm90_u8_items_per_thread             = 12;
inline constexpr int sm90_u16_threads_per_block           = 960;
inline constexpr int sm90_u16_items_per_thread            = 10;
inline constexpr int sm100_multi_threads_per_block        = 1024;
inline constexpr int sm100_multi_even_nominal_items       = 8;
inline constexpr int sm100_multi_range_nominal_items      = 16;
inline constexpr int sm100_u8_even_threads_per_block      = 928;
inline constexpr int sm100_u8_range_threads_per_block     = 448;
inline constexpr int sm100_u8_items_per_thread            = 12;
inline constexpr int sm100_u32_threads_per_block          = 768;
inline constexpr int sm100_u32_items_per_thread           = 12;
inline constexpr int sm100_u64_threads_per_block          = 768;
inline constexpr int sm100_u64_items_per_thread           = 6;
inline constexpr int sm100_range_static_threads_per_block = 384;
inline constexpr int sm100_range_u32_threads_per_block    = 768;
inline constexpr int sm100_range_u64_nominal_items        = 16;
inline constexpr int sm100_range_static_min_blocks_per_sm = 3;

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool supports_dependent_launch(::cuda::compute_capability cc)
{
  return cc >= sm90;
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

  const bool prefer_dynamic_smem = counter_size > supported_counter_bytes
                                || !should_use_static_smem(policy, num_bins, counter_size, num_active_channels);

  int max_bins = max_privatized_dynamic_smem_bins(policy, counter_size, num_active_channels);
  if (num_active_channels >= first_multi_channel_count)
  {
    if constexpr (IsEven)
    {
      max_bins = num_active_channels == two_active_channels   ? policy.dynamic_smem.even_2ch_max_bins
               : num_active_channels == three_active_channels ? policy.dynamic_smem.even_3ch_max_bins
               : num_active_channels == four_active_channels
                 ? policy.dynamic_smem.even_4ch_max_bins
                 : 0;
    }
    else
    {
      max_bins = policy.dynamic_smem.range_max_bins;
    }
  }

  return prefer_dynamic_smem && max_bins > 0 && num_bins <= max_bins;
}

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

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto get_sm50_tuning() const -> HistogramPolicy
  {
    const auto kernel = HistogramPolicy::Kernel{
      default_threads_per_block,
      t_scale(default_nominal_items_per_thread),
      default_vec_size,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      true,
      false};
    return {kernel,
            {kernel, pre_sm100_static_smem_max_bins * counter_size_bytes * num_active_channels, 0},
            {kernel, 0, 0, 0, 0, 0},
            0};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto get_sm90_tuning() const -> HistogramPolicy
  {
    auto result = get_sm50_tuning();
    if (num_channels == single_channel_count && num_active_channels == single_channel_count
        && counter_size_bytes == supported_counter_bytes && sample_is_primitive)
    {
      if (sample_size_bytes == sample_u8_bytes)
      {
        result.gmem = {
          sm90_u8_threads_per_block,
          sm90_u8_items_per_thread,
          default_vec_size,
          BLOCK_LOAD_DIRECT,
          LOAD_LDG,
          false,
          false};
      }
      else if (sample_size_bytes == sample_u16_bytes)
      {
        result.gmem = {
          sm90_u16_threads_per_block,
          sm90_u16_items_per_thread,
          default_vec_size,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          true,
          false};
      }
      result.static_smem.kernel  = result.gmem;
      result.dynamic_smem.kernel = result.gmem;
      result.init_kernel_pdl_trigger_max_bins =
        sample_size_bytes == sample_u8_bytes || sample_size_bytes == sample_u16_bytes ? pdl_trigger_max_bins : 0;
    }
    return result;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto get_sm100_tuning() const -> HistogramPolicy
  {
    auto result               = get_sm90_tuning();
    const bool single_channel = num_channels == single_channel_count && num_active_channels == single_channel_count;
    if (num_channels >= first_multi_channel_count && counter_size_bytes == supported_counter_bytes
        && sample_is_primitive)
    {
      result.gmem = {
        sm100_multi_threads_per_block,
        t_scale(is_even ? sm100_multi_even_nominal_items : sm100_multi_range_nominal_items),
        default_vec_size,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        true,
        false};
    }
    else if (single_channel && counter_size_bytes == supported_counter_bytes && sample_is_primitive)
    {
      if (sample_size_bytes == sample_u8_bytes)
      {
        result.gmem =
          is_even
            ? HistogramPolicy::Kernel{sm100_u8_even_threads_per_block,
                                      sm100_u8_items_per_thread,
                                      default_vec_size,
                                      BLOCK_LOAD_DIRECT,
                                      LOAD_CA,
                                      false,
                                      false}
            : HistogramPolicy::Kernel{
                sm100_u8_range_threads_per_block,
                sm100_u8_items_per_thread,
                default_vec_size,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                false,
                false};
      }
      else if (sample_size_bytes == sample_u32_bytes)
      {
        result.gmem = {
          sm100_u32_threads_per_block,
          sm100_u32_items_per_thread,
          default_vec_size,
          BLOCK_LOAD_DIRECT,
          LOAD_LDG,
          true,
          false};
      }
      else if (sample_size_bytes == sample_u64_bytes)
      {
        result.gmem = {
          sm100_u64_threads_per_block,
          sm100_u64_items_per_thread,
          default_vec_size,
          BLOCK_LOAD_DIRECT,
          LOAD_LDG,
          true,
          false};
      }
    }

    result.static_smem.kernel     = result.gmem;
    result.dynamic_smem.kernel    = result.gmem;
    const bool range_multi_static = !is_even && num_channels >= first_multi_channel_count
                                 && counter_size_bytes == supported_counter_bytes && sample_is_primitive;
    const bool range_u32_static = !is_even && single_channel && counter_size_bytes == supported_counter_bytes
                               && sample_is_primitive && sample_size_bytes == sample_u32_bytes;
    const bool range_u64_static = !is_even && single_channel && counter_size_bytes == supported_counter_bytes
                               && sample_is_primitive && sample_size_bytes == sample_u64_bytes;
    if (range_multi_static || range_u64_static)
    {
      result.static_smem.kernel.threads_per_block = sm100_range_static_threads_per_block;
    }
    else if (range_u32_static)
    {
      result.static_smem.kernel.threads_per_block = sm100_range_u32_threads_per_block;
    }
    if (range_u64_static)
    {
      result.static_smem.kernel.items_per_thread = t_scale(sm100_range_u64_nominal_items);
    }
    result.static_smem.max_privatized_smem_bytes =
      sm100_static_smem_max_bins * counter_size_bytes * num_active_channels;
    result.static_smem.min_blocks_per_sm =
      range_multi_static || range_u64_static ? sm100_range_static_min_blocks_per_sm : 0;

    const bool has_dynamic_smem_tuning =
      counter_size_bytes == supported_counter_bytes && sample_is_primitive
      && ((single_channel
           && (sample_size_bytes == sample_u8_bytes || sample_size_bytes == sample_u32_bytes
               || sample_size_bytes == sample_u64_bytes))
          || num_channels >= first_multi_channel_count);
    result.dynamic_smem = {
      result.gmem,
      has_dynamic_smem_tuning ? sm100_dynamic_smem_max_bytes : 0,
      has_dynamic_smem_tuning ? sm100_range_dynamic_smem_max_bins : 0,
      has_dynamic_smem_tuning ? sm100_even_2ch_dynamic_smem_max_bins : 0,
      has_dynamic_smem_tuning ? sm100_even_3ch_dynamic_smem_max_bins : 0,
      has_dynamic_smem_tuning ? sm100_even_4ch_dynamic_smem_max_bins : 0};
    result.init_kernel_pdl_trigger_max_bins =
      single_channel && counter_size_bytes == supported_counter_bytes && sample_is_primitive
          && (sample_size_bytes == sample_u8_bytes || sample_size_bytes == sample_u16_bytes
              || sample_size_bytes == sample_u32_bytes || sample_size_bytes == sample_u64_bytes)
        ? pdl_trigger_max_bins
        : 0;
    return result;
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    return cc >= sm100 ? get_sm100_tuning() : cc >= sm90 ? get_sm90_tuning() : get_sm50_tuning();
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
