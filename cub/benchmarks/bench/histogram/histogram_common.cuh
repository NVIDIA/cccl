// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_histogram.cuh>

#include <cuda/std/type_traits>

#if !TUNE_BASE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

#  define TUNE_VEC_SIZE (1 << TUNE_VEC_SIZE_POW)

#  if TUNE_MEM_PREFERENCE == 0
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::GMEM;
#  elif TUNE_MEM_PREFERENCE == 1
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::SMEM;
#  else // TUNE_MEM_PREFERENCE == 2
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::BLEND;
#  endif // TUNE_MEM_PREFERENCE

#  if TUNE_LOAD_ALGORITHM_ID == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  elif TUNE_LOAD_ALGORITHM_ID == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  else
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_STRIPED
#  endif // TUNE_LOAD_ALGORITHM_ID

template <typename SampleT, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<500, policy_t, policy_t>
  {
    static constexpr cub::BlockLoadAlgorithm load_algorithm =
      (TUNE_LOAD_ALGORITHM == cub::BLOCK_LOAD_STRIPED)
        ? (NUM_CHANNELS == 1 ? cub::BLOCK_LOAD_STRIPED : cub::BLOCK_LOAD_DIRECT)
        : TUNE_LOAD_ALGORITHM;

    using AgentHistogramPolicyT = cub::AgentHistogramPolicy<
      TUNE_THREADS,
      TUNE_ITEMS,
      load_algorithm,
      TUNE_LOAD_MODIFIER,
      TUNE_RLE_COMPRESS,
      MEM_PREFERENCE,
      TUNE_WORK_STEALING,
      TUNE_VEC_SIZE>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <class SampleT, class OffsetT>
SampleT get_upper_level(OffsetT bins, OffsetT elements)
{
  if constexpr (cuda::std::is_integral_v<SampleT>)
  {
    if constexpr (sizeof(SampleT) < sizeof(OffsetT))
    {
      const SampleT max_key = ::cuda::std::numeric_limits<SampleT>::max();
      return static_cast<SampleT>(std::min(bins, static_cast<OffsetT>(max_key)));
    }
    else
    {
      return static_cast<SampleT>(bins);
    }
  }

  return static_cast<SampleT>(elements);
}
