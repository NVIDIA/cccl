/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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

namespace detail
{
namespace histogram
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
  return Traits<T>::PRIMITIVE ? primitive_sample::yes : primitive_sample::no;
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
struct sm100_tuning<1, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 928;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_CA;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int tune_vec_size                             = 1 << 2;
};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{};

// range
template <class SampleT>
struct sm100_tuning<0, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 448;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int tune_vec_size                             = 1 << 2;
};

// same as base
template <class SampleT>
struct sm100_tuning<0, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{};

template <class SampleT>
struct sm100_tuning<0, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{
  // ipt_9.tpb_1024.rle_1.ws_0.mem_1.ld_0.laid_1.vec_0 1.358537  1.001009  1.373329  2.614104
  static constexpr int items                                     = 9;
  static constexpr int threads                                   = 1024;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int tune_vec_size                             = 1 << 0;
};

template <class SampleT>
struct sm100_tuning<0, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{
  // ipt_7.tpb_544.rle_1.ws_0.mem_1.ld_1.laid_0.vec_0 1.105331  0.934888  1.108557  1.391657
  static constexpr int items                                     = 7;
  static constexpr int threads                                   = 544;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int tune_vec_size                             = 1 << 0;
};

// multi.even
template <class SampleT>
struct sm100_tuning<1, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_9.tpb_1024.rle_0.ws_0.mem_1.ld_1.laid_1.vec_0 1.629591  0.997416  1.570900  2.772504
  static constexpr int items                                     = 9;
  static constexpr int threads                                   = 1024;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int tune_vec_size                             = 1 << 0;
};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_2>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_4>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{};

// same as base
template <class SampleT>
struct sm100_tuning<1, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_8>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{};

// multi.range
template <class SampleT>
struct sm100_tuning<0, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_7.tpb_160.rle_0.ws_0.mem_1.ld_1.laid_1.vec_1 1.210837  0.99556  1.189049  1.939584
  static constexpr int items                                     = 7;
  static constexpr int threads                                   = 160;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int tune_vec_size                             = 1 << 1;
};

// same as base
template <class SampleT>
struct sm100_tuning<0, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_2>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{};

// same as base
template <class SampleT>
struct sm100_tuning<0, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_4>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{};

// same as base
template <class SampleT>
struct sm100_tuning<0, SampleT, 4, 3, counter_size::_4, primitive_sample::yes, sample_size::_8>
    : sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{};

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  // TODO(bgruber): move inside t_scale in C++14
  static constexpr int v_scale = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);

  static constexpr int t_scale(int nominalItemsPerThread)
  {
    return (::cuda::std::max)(nominalItemsPerThread / NumActiveChannels / v_scale, 1);
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
  };

  using MaxPolicy = Policy1000;
};
} // namespace histogram
} // namespace detail

CUB_NAMESPACE_END
