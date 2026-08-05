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
  int gmem_threads_per_block;
  int gmem_items_per_thread;
  int gmem_vec_size;
  BlockLoadAlgorithm gmem_load_algorithm;
  CacheLoadModifier gmem_load_modifier;
  bool gmem_rle_compress;
  bool gmem_work_stealing;

  int static_smem_threads_per_block;
  int static_smem_items_per_thread;
  int static_smem_vec_size;
  BlockLoadAlgorithm static_smem_load_algorithm;
  CacheLoadModifier static_smem_load_modifier;
  bool static_smem_rle_compress;
  bool static_smem_work_stealing;
  int static_smem_max_privatized_bytes;
  int static_smem_min_blocks_per_sm;

  int dynamic_smem_threads_per_block;
  int dynamic_smem_items_per_thread;
  int dynamic_smem_vec_size;
  BlockLoadAlgorithm dynamic_smem_load_algorithm;
  CacheLoadModifier dynamic_smem_load_modifier;
  bool dynamic_smem_rle_compress;
  bool dynamic_smem_work_stealing;
  int dynamic_smem_max_privatized_bytes;
  int dynamic_smem_range_max_bins;
  int dynamic_smem_even_2ch_max_bins;
  int dynamic_smem_even_3ch_max_bins;
  int dynamic_smem_even_4ch_max_bins;

  int init_kernel_pdl_trigger_max_bins; //!< Common init-kernel PDL threshold, independent of accumulation tier

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const HistogramPolicy& lhs, const HistogramPolicy& rhs) noexcept
  {
    return lhs.gmem_threads_per_block == rhs.gmem_threads_per_block
        && lhs.gmem_items_per_thread == rhs.gmem_items_per_thread && lhs.gmem_vec_size == rhs.gmem_vec_size
        && lhs.gmem_load_algorithm == rhs.gmem_load_algorithm && lhs.gmem_load_modifier == rhs.gmem_load_modifier
        && lhs.gmem_rle_compress == rhs.gmem_rle_compress && lhs.gmem_work_stealing == rhs.gmem_work_stealing
        && lhs.static_smem_threads_per_block == rhs.static_smem_threads_per_block
        && lhs.static_smem_items_per_thread == rhs.static_smem_items_per_thread
        && lhs.static_smem_vec_size == rhs.static_smem_vec_size
        && lhs.static_smem_load_algorithm == rhs.static_smem_load_algorithm
        && lhs.static_smem_load_modifier == rhs.static_smem_load_modifier
        && lhs.static_smem_rle_compress == rhs.static_smem_rle_compress
        && lhs.static_smem_work_stealing == rhs.static_smem_work_stealing
        && lhs.static_smem_max_privatized_bytes == rhs.static_smem_max_privatized_bytes
        && lhs.static_smem_min_blocks_per_sm == rhs.static_smem_min_blocks_per_sm
        && lhs.dynamic_smem_threads_per_block == rhs.dynamic_smem_threads_per_block
        && lhs.dynamic_smem_items_per_thread == rhs.dynamic_smem_items_per_thread
        && lhs.dynamic_smem_vec_size == rhs.dynamic_smem_vec_size
        && lhs.dynamic_smem_load_algorithm == rhs.dynamic_smem_load_algorithm
        && lhs.dynamic_smem_load_modifier == rhs.dynamic_smem_load_modifier
        && lhs.dynamic_smem_rle_compress == rhs.dynamic_smem_rle_compress
        && lhs.dynamic_smem_work_stealing == rhs.dynamic_smem_work_stealing
        && lhs.dynamic_smem_max_privatized_bytes == rhs.dynamic_smem_max_privatized_bytes
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
        << "HistogramPolicy { .gmem_threads_per_block = " << p.gmem_threads_per_block
        << ", .gmem_items_per_thread = " << p.gmem_items_per_thread << ", .gmem_vec_size = " << p.gmem_vec_size
        << ", .gmem_load_algorithm = " << p.gmem_load_algorithm << ", .gmem_load_modifier = " << p.gmem_load_modifier
        << ", .gmem_rle_compress = " << p.gmem_rle_compress << ", .gmem_work_stealing = " << p.gmem_work_stealing
        << ", .static_smem_threads_per_block = " << p.static_smem_threads_per_block
        << ", .static_smem_items_per_thread = " << p.static_smem_items_per_thread << ", .static_smem_vec_size = "
        << p.static_smem_vec_size << ", .static_smem_load_algorithm = " << p.static_smem_load_algorithm
        << ", .static_smem_load_modifier = " << p.static_smem_load_modifier << ", .static_smem_rle_compress = "
        << p.static_smem_rle_compress << ", .static_smem_work_stealing = " << p.static_smem_work_stealing
        << ", .static_smem_max_privatized_bytes = " << p.static_smem_max_privatized_bytes
        << ", .static_smem_min_blocks_per_sm = " << p.static_smem_min_blocks_per_sm
        << ", .dynamic_smem_threads_per_block = " << p.dynamic_smem_threads_per_block
        << ", .dynamic_smem_items_per_thread = " << p.dynamic_smem_items_per_thread << ", .dynamic_smem_vec_size = "
        << p.dynamic_smem_vec_size << ", .dynamic_smem_load_algorithm = " << p.dynamic_smem_load_algorithm
        << ", .dynamic_smem_load_modifier = " << p.dynamic_smem_load_modifier << ", .dynamic_smem_rle_compress = "
        << p.dynamic_smem_rle_compress << ", .dynamic_smem_work_stealing = " << p.dynamic_smem_work_stealing
        << ", .dynamic_smem_max_privatized_bytes = " << p.dynamic_smem_max_privatized_bytes
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
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int threads_per_block(const HistogramPolicy& policy, PrivatizationMode)
{
  if constexpr (is_privatized_static_smem_v<PrivatizationMode>)
  {
    return policy.static_smem_threads_per_block;
  }
  else if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)
  {
    return policy.dynamic_smem_threads_per_block;
  }
  else
  {
    static_assert(is_privatized_gmem_v<PrivatizationMode>);
    return policy.gmem_threads_per_block;
  }
}

#define CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(NAME, TYPE)                                              \
  template <class PrivatizationMode>                                                                        \
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr TYPE NAME(const HistogramPolicy& policy, PrivatizationMode) \
  {                                                                                                         \
    if constexpr (is_privatized_static_smem_v<PrivatizationMode>)                                           \
    {                                                                                                       \
      return policy.static_smem_##NAME;                                                                     \
    }                                                                                                       \
    else if constexpr (is_privatized_dynamic_smem_v<PrivatizationMode>)                                     \
    {                                                                                                       \
      return policy.dynamic_smem_##NAME;                                                                    \
    }                                                                                                       \
    else                                                                                                    \
    {                                                                                                       \
      static_assert(is_privatized_gmem_v<PrivatizationMode>);                                               \
      return policy.gmem_##NAME;                                                                            \
    }                                                                                                       \
  }

CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(items_per_thread, int)
CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(vec_size, int)
CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(load_algorithm, BlockLoadAlgorithm)
CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(load_modifier, CacheLoadModifier)
CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(rle_compress, bool)
CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR(work_stealing, bool)

#undef CUB_DETAIL_HISTOGRAM_POLICY_FIELD_ACCESSOR

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
  return max_privatized_smem_bins(policy.static_smem_max_privatized_bytes, counter_size, num_active_channels);
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_privatized_dynamic_smem_bins(const HistogramPolicy& policy, int counter_size, int num_active_channels)
{
  return max_privatized_smem_bins(policy.dynamic_smem_max_privatized_bytes, counter_size, num_active_channels);
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
      max_bins = num_active_channels == 2 ? policy.dynamic_smem_even_2ch_max_bins
               : num_active_channels == 3
                 ? policy.dynamic_smem_even_3ch_max_bins
                 : policy.dynamic_smem_even_4ch_max_bins;
    }
    else
    {
      max_bins = policy.dynamic_smem_range_max_bins;
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
    using AgentHistogramPolicyT =
      legacy_agent_histogram_policy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, false>;
    static constexpr int init_kernel_pdl_trigger_max_bins = 0;
  };

  // SM90
  struct Policy900 : detail::chained_policy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> legacy_agent_histogram_policy<Tuning::threads_per_block,
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
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int) -> legacy_agent_histogram_policy<
      Tuning::threads_per_block,
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_policy(
    int threads_per_block,
    int items_per_thread,
    int vec_size,
    BlockLoadAlgorithm load_algorithm,
    CacheLoadModifier load_modifier,
    bool rle_compress,
    bool work_stealing,
    int static_smem_max_bytes,
    int pdl_max_bins) const -> HistogramPolicy
  {
    return {
      threads_per_block,
      items_per_thread,
      vec_size,
      load_algorithm,
      load_modifier,
      rle_compress,
      work_stealing,
      threads_per_block,
      items_per_thread,
      vec_size,
      load_algorithm,
      load_modifier,
      rle_compress,
      work_stealing,
      static_smem_max_bytes,
      0,
      threads_per_block,
      items_per_thread,
      vec_size,
      load_algorithm,
      load_modifier,
      rle_compress,
      work_stealing,
      0,
      0,
      0,
      0,
      0,
      pdl_max_bins};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto default_policy() const -> HistogramPolicy
  {
    return make_policy(
      384,
      t_scale(16),
      4,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      true,
      false,
      pre_sm100_static_smem_max_bins * counter_size_bytes * num_active_channels,
      0);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm90_policy() const -> HistogramPolicy
  {
    const bool single_channel = num_channels == 1 && num_active_channels == 1;
    const int pdl_max_bins =
      single_channel && counter_size_bytes == 4 && sample_is_primitive
          && (sample_size_bytes == 1 || sample_size_bytes == 2)
        ? pdl_trigger_max_bins
        : 0;
    if (single_channel && counter_size_bytes == 4 && sample_is_primitive)
    {
      if (sample_size_bytes == 1)
      {
        return make_policy(
          768,
          12,
          4,
          BLOCK_LOAD_DIRECT,
          LOAD_LDG,
          false,
          false,
          pre_sm100_static_smem_max_bins * counter_size_bytes * num_active_channels,
          pdl_max_bins);
      }
      if (sample_size_bytes == 2)
      {
        return make_policy(
          960,
          10,
          4,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          true,
          false,
          pre_sm100_static_smem_max_bins * counter_size_bytes * num_active_channels,
          pdl_max_bins);
      }
    }
    auto result                             = default_policy();
    result.init_kernel_pdl_trigger_max_bins = pdl_max_bins;
    return result;
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> HistogramPolicy
  {
    if (cc < ::cuda::compute_capability{9, 0})
    {
      return default_policy();
    }
    if (cc < ::cuda::compute_capability{10, 0})
    {
      return sm90_policy();
    }

    const bool single_channel = num_channels == 1 && num_active_channels == 1;
    auto result               = sm90_policy();
    if (num_channels >= 2 && counter_size_bytes == 4 && sample_is_primitive)
    {
      result.gmem_threads_per_block = 1024;
      result.gmem_items_per_thread  = t_scale(is_even ? 8 : 16);
    }
    else if (single_channel && counter_size_bytes == 4 && sample_is_primitive)
    {
      if (sample_size_bytes == 1)
      {
        result.gmem_threads_per_block = is_even ? 928 : 448;
        result.gmem_items_per_thread  = 12;
        result.gmem_load_modifier     = is_even ? LOAD_CA : LOAD_LDG;
        result.gmem_rle_compress      = false;
      }
      else if (sample_size_bytes == 4)
      {
        result.gmem_threads_per_block = 768;
        result.gmem_items_per_thread  = 12;
      }
      else if (sample_size_bytes == 8)
      {
        result.gmem_threads_per_block = 768;
        result.gmem_items_per_thread  = 6;
      }
    }

    result.static_smem_threads_per_block = result.gmem_threads_per_block;
    result.static_smem_items_per_thread  = result.gmem_items_per_thread;
    result.static_smem_vec_size          = result.gmem_vec_size;
    result.static_smem_load_algorithm    = result.gmem_load_algorithm;
    result.static_smem_load_modifier     = result.gmem_load_modifier;
    result.static_smem_rle_compress      = result.gmem_rle_compress;
    result.static_smem_work_stealing     = result.gmem_work_stealing;

    const bool range_multi_static = !is_even && num_channels >= 2 && counter_size_bytes == 4 && sample_is_primitive;
    const bool range_u32_static =
      !is_even && single_channel && counter_size_bytes == 4 && sample_is_primitive && sample_size_bytes == 4;
    const bool range_u64_static =
      !is_even && single_channel && counter_size_bytes == 4 && sample_is_primitive && sample_size_bytes == 8;
    if (range_multi_static || range_u64_static)
    {
      result.static_smem_threads_per_block = 384;
    }
    else if (range_u32_static)
    {
      result.static_smem_threads_per_block = 768;
    }
    if (range_u64_static)
    {
      result.static_smem_items_per_thread = t_scale(16);
    }
    result.static_smem_max_privatized_bytes = sm100_static_smem_max_bins * counter_size_bytes * num_active_channels;
    result.static_smem_min_blocks_per_sm    = range_multi_static || range_u64_static ? 3 : 0;

    result.dynamic_smem_threads_per_block = result.gmem_threads_per_block;
    result.dynamic_smem_items_per_thread  = result.gmem_items_per_thread;
    result.dynamic_smem_vec_size          = result.gmem_vec_size;
    result.dynamic_smem_load_algorithm    = result.gmem_load_algorithm;
    result.dynamic_smem_load_modifier     = result.gmem_load_modifier;
    result.dynamic_smem_rle_compress      = result.gmem_rle_compress;
    result.dynamic_smem_work_stealing     = result.gmem_work_stealing;

    const bool has_dynamic_smem_tuning =
      counter_size_bytes == 4 && sample_is_primitive
      && ((single_channel && (sample_size_bytes == 1 || sample_size_bytes == 4 || sample_size_bytes == 8))
          || num_channels >= 2);
    if (has_dynamic_smem_tuning)
    {
      result.dynamic_smem_max_privatized_bytes = sm100_dynamic_smem_max_bytes;
      result.dynamic_smem_range_max_bins       = sm100_range_dynamic_smem_max_bins;
      result.dynamic_smem_even_2ch_max_bins    = sm100_even_2ch_dynamic_smem_max_bins;
      result.dynamic_smem_even_3ch_max_bins    = sm100_even_3ch_dynamic_smem_max_bins;
      result.dynamic_smem_even_4ch_max_bins    = sm100_even_4ch_dynamic_smem_max_bins;
    }

    result.init_kernel_pdl_trigger_max_bins =
      single_channel && counter_size_bytes == 4 && sample_is_primitive
          && (sample_size_bytes == 1 || sample_size_bytes == 2 || sample_size_bytes == 4 || sample_size_bytes == 8)
        ? pdl_trigger_max_bins
        : 0;
    return result;
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
