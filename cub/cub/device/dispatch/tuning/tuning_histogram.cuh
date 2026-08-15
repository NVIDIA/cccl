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
  int max_privatized_static_smem_single_channel_bytes; //!< Single-channel compile-time-sized SMEM limit
  int max_privatized_dynamic_smem_single_channel_bytes; //!< Single-channel runtime-sized SMEM limit
  int static_smem_min_blocks_per_sm; //!< Minimum blocks per SM requested by the static-SMEM launch bounds
  int max_privatized_dynamic_smem_multi_channel_range_bytes; //!< Multi-channel HistogramRange SMEM limit
  int max_privatized_dynamic_smem_2_channel_even_bytes; //!< Two-channel HistogramEven SMEM limit
  int max_privatized_dynamic_smem_3_channel_even_bytes; //!< Three-channel HistogramEven SMEM limit
  int max_privatized_dynamic_smem_4_channel_even_bytes; //!< Four-channel HistogramEven SMEM limit
  int max_output_histogram_bytes_for_init_kernel_pdl_trigger; //!< Largest output allocation for init-kernel PDL

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.gmem == rhs.gmem && lhs.static_smem == rhs.static_smem && lhs.dynamic_smem == rhs.dynamic_smem
        && lhs.max_privatized_static_smem_single_channel_bytes == rhs.max_privatized_static_smem_single_channel_bytes
        && lhs.max_privatized_dynamic_smem_single_channel_bytes == rhs.max_privatized_dynamic_smem_single_channel_bytes
        && lhs.static_smem_min_blocks_per_sm == rhs.static_smem_min_blocks_per_sm
        && lhs.max_privatized_dynamic_smem_multi_channel_range_bytes
             == rhs.max_privatized_dynamic_smem_multi_channel_range_bytes
        && lhs.max_privatized_dynamic_smem_2_channel_even_bytes == rhs.max_privatized_dynamic_smem_2_channel_even_bytes
        && lhs.max_privatized_dynamic_smem_3_channel_even_bytes == rhs.max_privatized_dynamic_smem_3_channel_even_bytes
        && lhs.max_privatized_dynamic_smem_4_channel_even_bytes == rhs.max_privatized_dynamic_smem_4_channel_even_bytes
        && lhs.max_output_histogram_bytes_for_init_kernel_pdl_trigger
             == rhs.max_output_histogram_bytes_for_init_kernel_pdl_trigger;
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
        << ", .dynamic_smem = " << p.dynamic_smem << ", .max_privatized_static_smem_single_channel_bytes = "
        << p.max_privatized_static_smem_single_channel_bytes << ", .max_privatized_dynamic_smem_single_channel_bytes = "
        << p.max_privatized_dynamic_smem_single_channel_bytes << ", .static_smem_min_blocks_per_sm = "
        << p.static_smem_min_blocks_per_sm << ", .max_privatized_dynamic_smem_multi_channel_range_bytes = "
        << p.max_privatized_dynamic_smem_multi_channel_range_bytes
        << ", .max_privatized_dynamic_smem_2_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_2_channel_even_bytes
        << ", .max_privatized_dynamic_smem_3_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_3_channel_even_bytes
        << ", .max_privatized_dynamic_smem_4_channel_even_bytes = "
        << p.max_privatized_dynamic_smem_4_channel_even_bytes
        << ", .max_output_histogram_bytes_for_init_kernel_pdl_trigger = "
        << p.max_output_histogram_bytes_for_init_kernel_pdl_trigger << " }";
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
  int dynamic_smem_max_bytes = policy.max_privatized_dynamic_smem_single_channel_bytes;
  if constexpr (NumActiveChannels > 1)
  {
    if constexpr (IsEven)
    {
      dynamic_smem_max_bytes =
        NumActiveChannels == 2   ? policy.max_privatized_dynamic_smem_2_channel_even_bytes
        : NumActiveChannels == 3 ? policy.max_privatized_dynamic_smem_3_channel_even_bytes
        : NumActiveChannels == 4
          ? policy.max_privatized_dynamic_smem_4_channel_even_bytes
          : 0;
    }
    else
    {
      dynamic_smem_max_bytes = policy.max_privatized_dynamic_smem_multi_channel_range_bytes;
    }
  }
  return dynamic_smem_max_bytes;
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

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    // SM100 and newer use the autoresearch launch shapes and dynamic-SMEM byte budgets.
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      const bool single_channel = num_channels == 1 && num_active_channels == 1;
      auto gmem = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};

      // Multi-channel 32-bit-counter histograms use the wider SM100 sweep tuned by autoresearch.
      if (num_channels > 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive)
      {
        gmem =
          HistogramPrivatizationPolicy{1024, t_scale(is_even ? 8 : 16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
      }
      // Single-channel primitive samples with 32-bit counters use their per-sample-width tuning.
      else if (single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive)
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
      constexpr int max_privatized_static_smem_bytes                       = 1024;
      constexpr int max_privatized_dynamic_smem_single_channel_bytes       = 228352;
      constexpr int max_privatized_dynamic_smem_range_bytes_per_channel    = 8192;
      constexpr int max_privatized_dynamic_smem_2_channel_even_bytes       = 228352;
      constexpr int max_privatized_dynamic_smem_3_channel_even_bytes       = 228348;
      constexpr int max_privatized_dynamic_smem_4_channel_even_bytes       = 131072;
      constexpr int max_output_histogram_bytes_for_init_kernel_pdl_trigger = 8192;

      // Dynamic-SMEM tuning exists only for the type and channel combinations measured by autoresearch.
      const bool has_dynamic_smem_tuning =
        counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
        && ((single_channel && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8))
            || num_channels > 1);
      const int init_kernel_pdl_trigger_bytes =
        single_channel && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)} && sample_is_primitive
            && (sample_size_bytes == 1 || sample_size_bytes == 2 || sample_size_bytes == 4 || sample_size_bytes == 8)
          ? max_output_histogram_bytes_for_init_kernel_pdl_trigger
          : 0;

      return HistogramPolicy{
        gmem,
        static_smem,
        gmem,
        max_privatized_static_smem_bytes,
        has_dynamic_smem_tuning ? max_privatized_dynamic_smem_single_channel_bytes : 0,
        range_multi_static || range_u64_static ? 3 : 0,
        has_dynamic_smem_tuning ? max_privatized_dynamic_smem_range_bytes_per_channel * num_active_channels : 0,
        has_dynamic_smem_tuning ? max_privatized_dynamic_smem_2_channel_even_bytes : 0,
        has_dynamic_smem_tuning ? max_privatized_dynamic_smem_3_channel_even_bytes : 0,
        has_dynamic_smem_tuning ? max_privatized_dynamic_smem_4_channel_even_bytes : 0,
        init_kernel_pdl_trigger_bytes};
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
      constexpr int max_privatized_static_smem_bytes                       = 1024;
      constexpr int max_output_histogram_bytes_for_init_kernel_pdl_trigger = 8192;
      const int init_kernel_pdl_trigger_bytes =
        num_channels == 1 && num_active_channels == 1 && counter_size_bytes == int{sizeof(::cuda::std::uint32_t)}
            && sample_is_primitive && (sample_size_bytes == 1 || sample_size_bytes == 2)
          ? max_output_histogram_bytes_for_init_kernel_pdl_trigger
          : 0;
      return HistogramPolicy{
        sweep, sweep, sweep, max_privatized_static_smem_bytes, 0, 0, 0, 0, 0, 0, init_kernel_pdl_trigger_bytes};
    }

    // Architectures before SM90 use the longstanding generic histogram tuning.
    const auto sweep = HistogramPrivatizationPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, false};
    return HistogramPolicy{sweep, sweep, sweep, 1024, 0, 0, 0, 0, 0, 0, 0};
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
      IsEven}(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
