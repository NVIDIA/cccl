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

#include <cuda/__device/arch_id.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

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
_CCCL_API constexpr primitive_sample is_primitive_sample()
{
  return is_primitive<T>::value ? primitive_sample::yes : primitive_sample::no;
}

// TODO(bgruber): drop in CCCL 4.0
template <class CounterT>
_CCCL_API constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT>
_CCCL_API constexpr sample_size classify_sample_size()
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

  static constexpr bool rle_compress  = false;
  static constexpr bool work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads = 960;
  static constexpr int items   = 10;

  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

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
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 928;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
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
  static constexpr bool work_stealing                            = false;
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

  _CCCL_API static constexpr int t_scale(int nominalItemsPerThread)
  {
    return (::cuda::std::max) (nominalItemsPerThread / NumActiveChannels / v_scale, 1);
  }

  // SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      AgentHistogramPolicy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
  };

  // SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_API static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing>;

    template <typename Tuning>
    _CCCL_API static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    _CCCL_API static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing,
                              Tuning::vec_size>;

    template <typename Tuning>
    _CCCL_API static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  using MaxPolicy = Policy1000;
};

struct histogram_policy
{
  int block_threads;
  int pixels_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool rle_compress;
  BlockHistogramMemoryPreference mem_preference;
  bool work_stealing;
  int vec_size;
  int pdl_trigger_next_launch_in_init_kernel_max_bin_count;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const histogram_policy& lhs, const histogram_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.pixels_per_thread == rhs.pixels_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.rle_compress == rhs.rle_compress && lhs.mem_preference == rhs.mem_preference
        && lhs.work_stealing == rhs.work_stealing && lhs.vec_size == rhs.vec_size
        && lhs.pdl_trigger_next_launch_in_init_kernel_max_bin_count
             == rhs.pdl_trigger_next_launch_in_init_kernel_max_bin_count;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const histogram_policy& lhs, const histogram_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const histogram_policy& p)
  {
    return os
        << "histogram_policy { .block_threads = " << p.block_threads << ", .pixels_per_thread = " << p.pixels_per_thread
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .rle_compress = " << p.rle_compress << ", .mem_preference = " << p.mem_preference
        << ", .work_stealing = " << p.work_stealing << ", .vec_size = " << p.vec_size
        << ", .pdl_trigger_next_launch_in_init_kernel_max_bin_count = "
        << p.pdl_trigger_next_launch_in_init_kernel_max_bin_count << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, histogram_policy>;
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
  [[nodiscard]] _CCCL_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> histogram_policy
  {
    if (arch >= ::cuda::arch_id::sm_100)
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 1)
      {
        if (is_even)
        {
          // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
          return histogram_policy{928, 12, BLOCK_LOAD_DIRECT, LOAD_CA, false, SMEM, false, 1 << 2, 2048};
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return histogram_policy{448, 12, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 1 << 2, 2048};
        }
      }

      // sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks
      // multi.even and multi.range: none of the found tunings surpassed the SM90 tuning during verification benchmarks
    }

    if (arch >= ::cuda::arch_id::sm_90)
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return histogram_policy{768, 12, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 1 << 2, 2048};
        }
        else if (sample_size == 2)
        {
          return histogram_policy{960, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 1 << 2, 2048};
        }
      }
    }

    // fallback from SM50
    return histogram_policy{384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 4, 0};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(histogram_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> histogram_policy
  {
    constexpr auto policies = policy_selector{
      is_primitive_v<SampleT>,
      int{sizeof(SampleT)},
      int{sizeof(CounterT)},
      int{sizeof(SampleT)},
      NumChannels,
      NumActiveChannels,
      IsEven};
    return policies(arch);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
