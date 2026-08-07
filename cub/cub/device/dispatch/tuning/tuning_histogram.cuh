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

//! The tuning policy for one DeviceHistogram sweep pass.
struct HistogramSweepPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  int vec_size; //!< Vectorization size for loading samples
  BlockLoadAlgorithm load_algorithm; //!< Algorithm used for loading samples
  CacheLoadModifier load_modifier; //!< Cache modifier used for loading samples
  bool rle_compress; //!< Whether to locally run-length encode samples
  bool work_stealing; //!< Whether blocks dequeue tiles from a global work queue

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramSweepPolicy& lhs, const HistogramSweepPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.work_stealing == rhs.work_stealing;
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const HistogramSweepPolicy& p)
  {
    return os
        << "HistogramSweepPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .vec_size = " << p.vec_size
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .rle_compress = " << p.rle_compress << ", .work_stealing = " << p.work_stealing << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all DeviceHistogram sweep passes.
struct HistogramPolicy
{
  HistogramSweepPolicy gmem; //!< Policy for global-memory privatization
  HistogramSweepPolicy static_smem; //!< Policy for compile-time-sized shared-memory privatization
  HistogramSweepPolicy dynamic_smem; //!< Policy for runtime-sized shared-memory privatization
  int max_privatized_static_smem_bytes; //!< Maximum compile-time-sized shared-memory allocation
  int static_smem_min_blocks_per_sm; //!< Minimum blocks per SM requested by the static-SMEM launch bounds
  int max_privatized_dynamic_smem_bytes; //!< Maximum runtime-sized shared-memory allocation
  int dynamic_smem_range_max_bins; //!< Multi-channel HistogramRange limit, in bins per channel
  int dynamic_smem_even_2ch_max_bins; //!< Two-channel HistogramEven limit, in bins per channel
  int dynamic_smem_even_3ch_max_bins; //!< Three-channel HistogramEven limit, in bins per channel
  int dynamic_smem_even_4ch_max_bins; //!< Four-channel HistogramEven limit, in bins per channel
  int init_kernel_pdl_trigger_max_bins; //!< Common init-kernel PDL threshold, independent of accumulation tier

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.gmem == rhs.gmem && lhs.static_smem == rhs.static_smem && lhs.dynamic_smem == rhs.dynamic_smem
        && lhs.max_privatized_static_smem_bytes == rhs.max_privatized_static_smem_bytes
        && lhs.static_smem_min_blocks_per_sm == rhs.static_smem_min_blocks_per_sm
        && lhs.max_privatized_dynamic_smem_bytes == rhs.max_privatized_dynamic_smem_bytes
        && lhs.dynamic_smem_range_max_bins == rhs.dynamic_smem_range_max_bins
        && lhs.dynamic_smem_even_2ch_max_bins == rhs.dynamic_smem_even_2ch_max_bins
        && lhs.dynamic_smem_even_3ch_max_bins == rhs.dynamic_smem_even_3ch_max_bins
        && lhs.dynamic_smem_even_4ch_max_bins == rhs.dynamic_smem_even_4ch_max_bins
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
    return os
        << "HistogramPolicy { .gmem = " << p.gmem << ", .static_smem = " << p.static_smem << ", .dynamic_smem = "
        << p.dynamic_smem << ", .max_privatized_static_smem_bytes = " << p.max_privatized_static_smem_bytes
        << ", .static_smem_min_blocks_per_sm = " << p.static_smem_min_blocks_per_sm
        << ", .max_privatized_dynamic_smem_bytes = " << p.max_privatized_dynamic_smem_bytes
        << ", .dynamic_smem_range_max_bins = " << p.dynamic_smem_range_max_bins
        << ", .dynamic_smem_even_2ch_max_bins = " << p.dynamic_smem_even_2ch_max_bins
        << ", .dynamic_smem_even_3ch_max_bins = " << p.dynamic_smem_even_3ch_max_bins
        << ", .dynamic_smem_even_4ch_max_bins = " << p.dynamic_smem_even_4ch_max_bins
        << ", .init_kernel_pdl_trigger_max_bins = " << p.init_kernel_pdl_trigger_max_bins << " }";
  }
#endif
};

namespace detail::histogram
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_privatized_smem_bins(int max_privatized_smem_bytes, int counter_size, int num_active_channels)
{
  if (max_privatized_smem_bytes <= 0 || counter_size <= 0 || num_active_channels <= 0)
  {
    return 0;
  }
  return max_privatized_smem_bytes / counter_size / num_active_channels;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool
should_use_static_smem(const HistogramPolicy& policy, int num_bins, int counter_size, int num_active_channels)
{
  return num_bins > 0
      && num_bins
           <= max_privatized_smem_bins(policy.max_privatized_static_smem_bytes, counter_size, num_active_channels);
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

  int max_bins = max_privatized_smem_bins(policy.max_privatized_dynamic_smem_bytes, counter_size, num_active_channels);
  if (num_active_channels > 1)
  {
    if constexpr (IsEven)
    {
      max_bins = num_active_channels == 2 ? policy.dynamic_smem_even_2ch_max_bins
               : num_active_channels == 3 ? policy.dynamic_smem_even_3ch_max_bins
               : num_active_channels == 4
                 ? policy.dynamic_smem_even_4ch_max_bins
                 : 0;
    }
    else
    {
      max_bins = policy.dynamic_smem_range_max_bins;
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

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      const bool single_channel = num_channels == 1 && num_active_channels == 1;
      auto gmem                 = HistogramSweepPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};

      if (num_channels > 1 && counter_size_bytes == int{sizeof(unsigned int)} && sample_is_primitive)
      {
        gmem = HistogramSweepPolicy{1024, t_scale(is_even ? 8 : 16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      }
      else if (single_channel && counter_size_bytes == int{sizeof(unsigned int)} && sample_is_primitive)
      {
        if (sample_size_bytes == 1)
        {
          gmem = is_even ? HistogramSweepPolicy{928, 12, 4, BLOCK_LOAD_DIRECT, LOAD_CA, false, false}
                         : HistogramSweepPolicy{448, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
        }
        else if (sample_size_bytes == 2)
        {
          // Retain the SM90 U16 sweep shape on SM100.
          gmem = HistogramSweepPolicy{960, 10, 4, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false};
        }
        else if (sample_size_bytes == 4)
        {
          gmem = HistogramSweepPolicy{768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
        }
        else if (sample_size_bytes == 8)
        {
          gmem = HistogramSweepPolicy{768, 6, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
        }
      }

      auto static_smem = gmem;
      const bool range_multi_static =
        !is_even && num_channels > 1 && counter_size_bytes == int{sizeof(unsigned int)} && sample_is_primitive;
      const bool range_u32_static = !is_even && single_channel && counter_size_bytes == int{sizeof(unsigned int)}
                                 && sample_is_primitive && sample_size_bytes == 4;
      const bool range_u64_static = !is_even && single_channel && counter_size_bytes == int{sizeof(unsigned int)}
                                 && sample_is_primitive && sample_size_bytes == 8;
      if (range_multi_static || range_u64_static)
      {
        static_smem.threads_per_block = 384;
      }
      else if (range_u32_static)
      {
        static_smem.threads_per_block = 768;
      }
      if (range_u64_static)
      {
        static_smem.items_per_thread = t_scale(16);
      }

      const bool has_dynamic_smem_tuning =
        counter_size_bytes == int{sizeof(unsigned int)} && sample_is_primitive
        && ((single_channel && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8))
            || num_channels > 1);
      // B200 provides 232448 bytes of opt-in shared memory. Reserve 4096 bytes
      // for the kernel's statically allocated shared-memory state.
      const int max_privatized_dynamic_smem_bytes = has_dynamic_smem_tuning ? 232448 - 4096 : 0;
      const int init_kernel_pdl_trigger_max_bins =
        single_channel && counter_size_bytes == int{sizeof(unsigned int)} && sample_is_primitive
            && (sample_size_bytes == 1 || sample_size_bytes == 2 || sample_size_bytes == 4 || sample_size_bytes == 8)
          ? 2048
          : 0;

      return HistogramPolicy{
        gmem,
        static_smem,
        gmem,
        512 * counter_size_bytes * num_active_channels,
        range_multi_static || range_u64_static ? 3 : 0,
        max_privatized_dynamic_smem_bytes,
        has_dynamic_smem_tuning ? 2048 : 0,
        has_dynamic_smem_tuning ? 28544 : 0,
        has_dynamic_smem_tuning ? 19029 : 0,
        has_dynamic_smem_tuning ? 8192 : 0,
        init_kernel_pdl_trigger_max_bins};
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      auto sweep = HistogramSweepPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      if (num_channels == 1 && num_active_channels == 1 && counter_size_bytes == int{sizeof(unsigned int)}
          && sample_is_primitive)
      {
        if (sample_size_bytes == 1)
        {
          sweep = HistogramSweepPolicy{768, 12, 4, BLOCK_LOAD_DIRECT, LOAD_LDG, false, false};
        }
        else if (sample_size_bytes == 2)
        {
          sweep = HistogramSweepPolicy{960, 10, 4, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, false};
        }
      }
      const int init_kernel_pdl_trigger_max_bins =
        num_channels == 1 && num_active_channels == 1 && counter_size_bytes == int{sizeof(unsigned int)}
            && sample_is_primitive && (sample_size_bytes == 1 || sample_size_bytes == 2)
          ? 2048
          : 0;
      return HistogramPolicy{
        sweep,
        sweep,
        sweep,
        256 * counter_size_bytes * num_active_channels,
        0,
        0,
        0,
        0,
        0,
        0,
        init_kernel_pdl_trigger_max_bins};
    }

    const auto sweep = HistogramSweepPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
    return HistogramPolicy{sweep, sweep, sweep, 256 * counter_size_bytes * num_active_channels, 0, 0, 0, 0, 0, 0, 0};
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
