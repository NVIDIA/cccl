// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_topk.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_type.cuh>

#include <cuda/atomic>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_topk
{
template <int BlockThreads, int ItemsPerThread>
struct AgentSegmentedTopkWorkerPerSegmentPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = BlockThreads;

  /// Items per thread (per tile of input)
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
};

template <typename ActivePolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
struct AgentSegmentedTopkWorkerPerSegment
{
  // -------------------------------------------------------------------------
  // Types and Constants
  // -------------------------------------------------------------------------
  // Derive inner types from Iterator of Iterators
  using key_it_t   = typename ::cuda::std::iterator_traits<KeyInputItItT>::value_type;
  using value_it_t = typename ::cuda::std::iterator_traits<ValueInputItItT>::value_type;

  using key_t   = typename ::cuda::std::iterator_traits<key_it_t>::value_type;
  using value_t = typename ::cuda::std::iterator_traits<value_it_t>::value_type;

  static constexpr int block_threads    = ActivePolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread = ActivePolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_size        = block_threads * items_per_thread;

  // Check if we are dealing with keys-only or key-value pairs
  static constexpr bool is_keys_only = ::cuda::std::is_same<value_t, cub::NullType>::value;

  // -------------------------------------------------------------------------
  // Primitive Types
  // -------------------------------------------------------------------------
  using BlockLoadKeysT = BlockLoad<key_t, block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadValsT = BlockLoad<value_t, block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE>;

  using BlockTopkT = BlockTopK<key_t, block_threads, items_per_thread, value_t>;

  using BlockStoreKeysT = BlockStore<key_t, block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreValsT = BlockStore<value_t, block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE>;

  // -------------------------------------------------------------------------
  // Shared Memory Storage
  // -------------------------------------------------------------------------
  struct TempStorage
  {
    union
    {
      typename BlockLoadKeysT::TempStorage load_keys;
      typename BlockLoadValsT::TempStorage load_vals;
      typename BlockTopkT::TempStorage topk;
      typename BlockStoreKeysT::TempStorage store_keys;
      typename BlockStoreValsT::TempStorage store_vals;
    };
  };

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSegmentedTopkWorkerPerSegment(
    TempStorage& temp_storage,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
      : temp_storage(temp_storage)
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , select_directions(select_directions)
      , num_segments(num_segments)
  {}

  // -------------------------------------------------------------------------
  // Processing Logic
  // -------------------------------------------------------------------------
  template <typename KValueT, detail::topk::select Direction>
  _CCCL_DEVICE _CCCL_FORCEINLINE void select_topk_keys(
    key_t (&keys)[items_per_thread], KValueT k, ::cuda::std::integral_constant<detail::topk::select, Direction>)
  {
    if constexpr (Direction == detail::topk::select::max)
    {
      BlockTopkT(temp_storage.topk).Max(keys, k);
    }
    else
    {
      BlockTopkT(temp_storage.topk).Min(keys, k);
    }
  }

  template <typename KValueT, detail::topk::select Direction>
  _CCCL_DEVICE _CCCL_FORCEINLINE void select_topk_pairs(
    key_t (&keys)[items_per_thread],
    value_t (&values)[items_per_thread],
    KValueT k,
    ::cuda::std::integral_constant<detail::topk::select, Direction>)
  {
    if constexpr (Direction == detail::topk::select::max)
    {
      BlockTopkT(temp_storage.topk).Max(keys, values, k);
    }
    else
    {
      BlockTopkT(temp_storage.topk).Min(keys, values, k);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    // Identify Segment
    int segment_id = blockIdx.x;

    // Boundary check
    if (segment_id >= resolve_param(num_segments, 0))
    {
      return;
    }

    // Resolve Segment Parameters
    const auto segment_size = resolve_param(segment_sizes, segment_id);
    const auto k            = resolve_param(k_param, segment_id);
    const auto direction    = resolve_param(select_directions, segment_id);

    // Determine padding key based on direction
    key_t padding_key = (direction == detail::topk::select::max)
                        ? ::cuda::std::numeric_limits<key_t>::lowest()
                        : ::cuda::std::numeric_limits<key_t>::max();

    // Dereference iterator-of-iterators to get the segment specific iterator
    auto block_keys_in = d_key_segments_it[segment_id];

    // Load Keys
    key_t thread_keys[items_per_thread];
    BlockLoadKeysT(temp_storage.load_keys).Load(block_keys_in, thread_keys, segment_size, padding_key);

    // Load Values (if applicable)
    [[maybe_unused]] value_t thread_values[items_per_thread];

    if constexpr (!is_keys_only)
    {
      __syncthreads();
      auto block_vals_in = d_value_segments_it[segment_id];

      BlockLoadValsT(temp_storage.load_vals).Load(block_vals_in, thread_values, segment_size);
    }

    __syncthreads();

    // Perform Block Top-K
    if constexpr (!is_keys_only)
    {
      // Pass both keys and values
      bool is_successful_dispatch = detail::params::dispatch_discrete(
        select_directions, segment_id, [this, &thread_keys, &thread_values, k](auto direction_tag) {
          select_topk_pairs(thread_keys, thread_values, k, direction_tag);
        });
      _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
    }
    else
    {
      bool is_successful_dispatch =
        detail::params::dispatch_discrete(select_directions, segment_id, [this, &thread_keys, k](auto direction_tag) {
          select_topk_keys(thread_keys, k, direction_tag);
        });
      _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
    }

    __syncthreads();

    auto block_keys_out = d_key_segments_out_it[segment_id];

    BlockStoreKeysT(temp_storage.store_keys)
      .Store(block_keys_out,
             thread_keys,
             k // Only store K items
      );

    if constexpr (!is_keys_only)
    {
      __syncthreads();
      auto block_vals_out = d_value_segments_out_it[segment_id];

      BlockStoreValsT(temp_storage.store_vals).Store(block_vals_out, thread_values, k);
    }
  }
};
} // namespace detail::segmented_topk
CUB_NAMESPACE_END
