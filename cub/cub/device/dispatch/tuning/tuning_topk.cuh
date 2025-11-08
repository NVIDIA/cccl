// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_load.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__algorithm/clamp.h>

CUB_NAMESPACE_BEGIN
namespace detail::topk
{
template <class KeyT>
constexpr int calc_bits_per_pass()
{
  switch (sizeof(KeyT))
  {
    case 1:
    default:
      return 8;
    case 2:
    case 4:
    case 8:
      return 11;
  }
}

template <class KeyInT>
struct sm90_tuning
{
  static constexpr int threads = 512; // Number of threads per block

  static constexpr int nominal_4b_items_per_thread = 4;
  static constexpr int items =
    ::cuda::std::max(1, (nominal_4b_items_per_thread * 4 / static_cast<int>(sizeof(KeyInT))));
  // Try to load 16 Bytes per thread. (int64(items=2);int32(items=4);int16(items=8)).

  static constexpr int bits_per_pass = calc_bits_per_pass<KeyInT>();

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_VECTORIZE;
};

template <class KeyInT, class OffsetT>
struct policy_hub
{
  struct DefaultTuning
  {
    static constexpr int nominal_4b_items_per_thread = 4;
    static constexpr int items_per_thread            = ::cuda::std::clamp(
      nominal_4b_items_per_thread * 4 / static_cast<int>(sizeof(KeyInT)), 1, nominal_4b_items_per_thread);
    static constexpr int bits_per_pass = calc_bits_per_pass<KeyInT>();

    using topk_policy_t =
      AgentTopKPolicy<512, items_per_thread, bits_per_pass, BLOCK_LOAD_VECTORIZE, BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy500
      : DefaultTuning
      , ChainedPolicy<350, Policy500, Policy500>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy500>
  {
    using tuning = sm90_tuning<KeyInT>;
    using topk_policy_t =
      AgentTopKPolicy<tuning::threads, tuning::items, tuning::bits_per_pass, tuning::load_algorithm, BLOCK_SCAN_WARP_SCANS>;
  };

  using max_policy = Policy900;
};
} // namespace detail::topk
CUB_NAMESPACE_END
