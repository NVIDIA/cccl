// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <int ItemsPerThread>
class block_topk_key_states;

//! @brief Block-level rank specialization that uses atomic counters for the
//!        final tie-break.
//!
//! The implementation is non-deterministic across launches because the tie-breaking subset is
//! chosen by `atomicAdd` ordering.
template <int BlockDimX>
class block_topk_rank_atomic
{
private:
  // TODO (elstehle): Make this configurable
  // Whether to include all items tied with the k-th key when selecting top-k
  static constexpr bool expand_k_to_include_ties = false;

  static constexpr int threads_per_block = BlockDimX;

  using counter_t = ::cuda::std::uint32_t;

  struct TempStorage_
  {
    counter_t selected_offset[2];
  };

  TempStorage_& storage;
  int linear_tid;

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit block_topk_rank_atomic(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(threads_per_block, 1, 1))
  {}

  //! Final tie-break + scatter-rank emission. See `block_topk_rank::rank_key_states`.
  template <int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  rank_key_states(block_topk_key_states<ItemsPerThread>& states, int (&scatter_ranks)[ItemsPerThread])
  {
    // Scatter indices of selected items into shared memory (only for selecting key-value pairs, using a two-phase
    // approach to lower shared memory requirements).
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      scatter_ranks[i] = -1;
    }

    // If all candidates are amongst the remaining top-k, we can simply select all non-rejected/non-invalid candidates.
    // Otherwise, we have to make sure that *all* selected candidates come first, and then select amongst tied
    // candidates to fill up the remaining slots up to k.
    const bool select_all_candidates = expand_k_to_include_ties || !states.has_ties();

    if (linear_tid == 0)
    {
      // Write offsets for selected items with key_prefix < kth_prefix
      storage.selected_offset[0] = counter_t{0};
      // Write offsets for tied items across the k-th position, i.e., key_prefix == kth_prefix
      storage.selected_offset[1] = static_cast<counter_t>(states.num_selected());
    }
    // Ensure atomic selection counter has been reset
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const bool is_selected  = states.is_selected(i);
      const bool is_candidate = states.is_candidate(i);

      // We differentiate between candidates and selected only if not all candidates make it into the top-k items.
      int item_class = (!select_all_candidates) && is_candidate ? 1 : 0;

      if (is_selected || is_candidate)
      {
        const auto selected_offset = static_cast<int>(atomicAdd(&storage.selected_offset[item_class], counter_t{1}));
        if (selected_offset < states.k())
        {
          scatter_ranks[i] = selected_offset;
          states.set_selected(i);
        }
        else
        {
          scatter_ranks[i] = -1;
          states.set_rejected(i);
        }
      }
    }
    states.set_num_selected(states.k());
    states.set_num_candidates(0);
  }
};
} // namespace detail

CUB_NAMESPACE_END
