// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
template <int BlockThreads, int ItemsPerThread>
struct policy_t
{
  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;
};

template <class PolicyT, class OffsetT, class OpT>
struct agent_block_striped_t
{
  static constexpr int items_per_thread = PolicyT::items_per_thread;

  OffsetT tile_base;
  OpT op;

  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(int items_in_tile, int block_threads)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; item++)
    {
      const auto idx = static_cast<OffsetT>(block_threads * item + threadIdx.x);

      if (IsFullTile || idx < items_in_tile)
      {
        (void) op(tile_base + idx);
      }
    }
  }
};
} // namespace detail::for_each

CUB_NAMESPACE_END
