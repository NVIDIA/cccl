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
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
enum class primitive_sample
{
  no,
  yes
};

enum class sample_size
{
  _1,
  _2,
  _4,
  _8,
  unknown
};

enum class counter_size
{
  _4,
  unknown
};

template <class T>
constexpr primitive_sample is_primitive_sample()
{
  return is_primitive<T>::value ? primitive_sample::yes : primitive_sample::no;
}

template <class CounterT>
constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

template <class SampleT>
constexpr sample_size classify_sample_size()
{
  return sizeof(SampleT) == 1 ? sample_size::_1 : sizeof(SampleT) == 2 ? sample_size::_2 : sample_size::unknown;
}

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

template <typename PolicyT, typename = void>
struct HistogramPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE HistogramPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct HistogramPolicyWrapper<StaticPolicyT,
                              ::cuda::std::void_t<decltype(StaticPolicyT::AgentHistogramPolicyT::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE HistogramPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr auto Histogram()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::AgentHistogramPolicyT());
  }

  _CCCL_HOST_DEVICE static constexpr int BlockThreads()
  {
    return StaticPolicyT::AgentHistogramPolicyT::BLOCK_THREADS;
  }

  _CCCL_HOST_DEVICE static constexpr int PixelsPerThread()
  {
    return StaticPolicyT::AgentHistogramPolicyT::PIXELS_PER_THREAD;
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"HistogramPolicy">() = Histogram().EncodedPolicy()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE HistogramPolicyWrapper<PolicyT> MakeHistogramPolicyWrapper(PolicyT policy)
{
  return HistogramPolicyWrapper<PolicyT>{policy};
}

// sample_size 2/4/8 showed no benefit over SM90 during verification benchmarks

// multi.even and multi.range: none of the found tunings surpassed the SM90 tuning during verification benchmarks

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  // TODO(bgruber): move inside t_scale in C++14
  static constexpr int v_scale = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);

  static constexpr int t_scale(int nominalItemsPerThread)
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
    static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing>;

    template <typename Tuning>
    static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing,
                              Tuning::vec_size>;

    template <typename Tuning>
    static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  using MaxPolicy = Policy1000;
};
} // namespace detail::histogram

CUB_NAMESPACE_END
