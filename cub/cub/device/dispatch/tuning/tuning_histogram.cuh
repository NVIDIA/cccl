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
  int dynamic_smem_bytes             = 0; //!< Tuned byte budget for a runtime-sized privatized histogram; 0 disables it
  int static_smem_threads_per_block  = 0; //!< Static shared-memory tier threads; 0 inherits threads_per_block
  int static_smem_items_per_thread   = 0; //!< Static shared-memory tier items; 0 inherits pixels_per_thread
  int static_smem_min_blocks_per_sm  = 0; //!< Static shared-memory launch bound; 0 derives it from the block size
  int dynamic_smem_range_max_bins    = 0; //!< Multi-channel RANGE cap per channel; 0 disables the dynamic path
  int dynamic_smem_even_2ch_max_bins = 0; //!< Two-channel EVEN cap per channel; 0 disables the dynamic path
  int dynamic_smem_even_3ch_max_bins = 0; //!< Three-channel EVEN cap per channel; 0 disables the dynamic path
  int dynamic_smem_even_4ch_max_bins = 0; //!< Four-channel EVEN cap per channel; 0 disables the dynamic path
  int range_interpolation_min_bins   = 0; //!< Minimum RANGE bin count for interpolation; 0 disables interpolation

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int static_smem_threads() const
  {
    return static_smem_threads_per_block != 0 ? static_smem_threads_per_block : threads_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int static_smem_items() const
  {
    return static_smem_items_per_thread != 0 ? static_smem_items_per_thread : pixels_per_thread;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int static_smem_min_blocks() const
  {
    return static_smem_min_blocks_per_sm != 0 ? static_smem_min_blocks_per_sm : (static_smem_threads() >= 512 ? 2 : 0);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.pixels_per_thread == rhs.pixels_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rle_compress == rhs.rle_compress
        && lhs.mem_preference == rhs.mem_preference && lhs.use_work_stealing == rhs.use_work_stealing
        && lhs.init_kernel_pdl_trigger_max_bins == rhs.init_kernel_pdl_trigger_max_bins
        && lhs.dynamic_smem_bytes == rhs.dynamic_smem_bytes
        && lhs.static_smem_threads_per_block == rhs.static_smem_threads_per_block
        && lhs.static_smem_items_per_thread == rhs.static_smem_items_per_thread
        && lhs.static_smem_min_blocks_per_sm == rhs.static_smem_min_blocks_per_sm
        && lhs.dynamic_smem_range_max_bins == rhs.dynamic_smem_range_max_bins
        && lhs.dynamic_smem_even_2ch_max_bins == rhs.dynamic_smem_even_2ch_max_bins
        && lhs.dynamic_smem_even_3ch_max_bins == rhs.dynamic_smem_even_3ch_max_bins
        && lhs.dynamic_smem_even_4ch_max_bins == rhs.dynamic_smem_even_4ch_max_bins
        && lhs.range_interpolation_min_bins == rhs.range_interpolation_min_bins;
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
        << ", .init_kernel_pdl_trigger_max_bins = " << p.init_kernel_pdl_trigger_max_bins << ", .dynamic_smem_bytes = "
        << p.dynamic_smem_bytes << ", .static_smem_threads_per_block = " << p.static_smem_threads_per_block
        << ", .static_smem_items_per_thread = " << p.static_smem_items_per_thread
        << ", .static_smem_min_blocks_per_sm = " << p.static_smem_min_blocks_per_sm
        << ", .dynamic_smem_range_max_bins = " << p.dynamic_smem_range_max_bins
        << ", .dynamic_smem_even_2ch_max_bins = " << p.dynamic_smem_even_2ch_max_bins
        << ", .dynamic_smem_even_3ch_max_bins = " << p.dynamic_smem_even_3ch_max_bins
        << ", .dynamic_smem_even_4ch_max_bins = " << p.dynamic_smem_even_4ch_max_bins
        << ", .range_interpolation_min_bins = " << p.range_interpolation_min_bins << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::histogram
{
// Leave 4096 bytes of the SM100 opt-in shared-memory limit available for static storage.
static constexpr int sm100_dynamic_smem_bytes             = 232448 - 4096;
static constexpr int sm100_dynamic_smem_range_max_bins    = 2048;
static constexpr int sm100_dynamic_smem_even_2ch_max_bins = 28544;
static constexpr int sm100_dynamic_smem_even_3ch_max_bins = 19029;
static constexpr int sm100_dynamic_smem_even_4ch_max_bins = 8192;
static constexpr int sm100_range_interpolation_min_bins   = 512;

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

template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 768;
  static constexpr bool rle_compress                             = true;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{
  static constexpr int items                                     = 6;
  static constexpr int threads                                   = 768;
  static constexpr bool rle_compress                             = true;
  static constexpr bool use_work_stealing                        = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// sample_size 2 retains the SM90 launch shape while using the SM100 shared-memory policy.

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

    using SelectedAgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    using MultiChannelAgentHistogramPolicyT =
      agent_histogram_policy<1024, t_scale(IsEven ? 8 : 16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 4>;

    static constexpr bool use_sm100_multi_channel_policy =
      NumChannels >= 2 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value;

    using AgentHistogramPolicyT =
      ::cuda::std::_If<use_sm100_multi_channel_policy, MultiChannelAgentHistogramPolicyT, SelectedAgentHistogramPolicyT>;

    static constexpr int init_kernel_pdl_trigger_max_bins =
      NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
          && (sizeof(SampleT) == 1 || sizeof(SampleT) == 2 || sizeof(SampleT) == 4 || sizeof(SampleT) == 8)
        ? 2048
        : 0;
    static constexpr int dynamic_smem_bytes             = sm100_dynamic_smem_bytes;
    static constexpr int dynamic_smem_range_max_bins    = sm100_dynamic_smem_range_max_bins;
    static constexpr int dynamic_smem_even_2ch_max_bins = sm100_dynamic_smem_even_2ch_max_bins;
    static constexpr int dynamic_smem_even_3ch_max_bins = sm100_dynamic_smem_even_3ch_max_bins;
    static constexpr int dynamic_smem_even_4ch_max_bins = sm100_dynamic_smem_even_4ch_max_bins;
    static constexpr int range_interpolation_min_bins   = sm100_range_interpolation_min_bins;
    static constexpr int static_smem_threads_per_block =
      !IsEven && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
        ? (NumChannels >= 2
             ? 384
             : (NumChannels == 1 && NumActiveChannels == 1 && sizeof(SampleT) == 4
                  ? 768
                  : (NumChannels == 1 && NumActiveChannels == 1 && sizeof(SampleT) == 8 ? 384 : 0)))
        : 0;
    static constexpr int static_smem_items_per_thread =
      !IsEven && NumChannels == 1 && NumActiveChannels == 1 && sizeof(CounterT) == 4 && is_primitive<SampleT>::value
          && sizeof(SampleT) == 8
        ? t_scale(16)
        : 0;
    static constexpr int static_smem_min_blocks_per_sm =
      !IsEven && static_smem_threads_per_block > 0 && static_smem_threads_per_block < 512 ? 3 : 0;
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm100_policy(HistogramPolicy policy) const -> HistogramPolicy
  {
    policy.dynamic_smem_bytes             = sm100_dynamic_smem_bytes;
    policy.dynamic_smem_range_max_bins    = sm100_dynamic_smem_range_max_bins;
    policy.dynamic_smem_even_2ch_max_bins = sm100_dynamic_smem_even_2ch_max_bins;
    policy.dynamic_smem_even_3ch_max_bins = sm100_dynamic_smem_even_3ch_max_bins;
    policy.dynamic_smem_even_4ch_max_bins = sm100_dynamic_smem_even_4ch_max_bins;
    policy.range_interpolation_min_bins   = sm100_range_interpolation_min_bins;
    return policy;
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
          return sm100_policy(HistogramPolicy{928, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_CA, false, SMEM, false, 2048});
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return sm100_policy(HistogramPolicy{448, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 2048});
        }
      }

      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive
          && (sample_size == 4 || sample_size == 8))
      {
        if (is_even)
        {
          return sm100_policy(
            HistogramPolicy{768, t_scale(12), 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 2048});
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
          SMEM,
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
          return sm100_policy(HistogramPolicy{1024, t_scale(8), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 0});
        }
        return sm100_policy(
          HistogramPolicy{1024, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 0, 0, 384, 0, 3});
      }

      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 2)
      {
        return sm100_policy(HistogramPolicy{960, 10, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 2048});
      }

      // Even when no SM100 launch-shape specialization applies, retain the
      // architecture's dynamic shared-memory budget on the inherited fallback.
      return sm100_policy(HistogramPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 0});
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return HistogramPolicy{768, 12, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 2048};
        }
        else if (sample_size == 2)
        {
          return HistogramPolicy{960, 10, 1 << 2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 2048};
        }
      }
    }

    // fallback from SM50
    return HistogramPolicy{384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 0};
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
