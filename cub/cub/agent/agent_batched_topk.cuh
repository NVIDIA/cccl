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

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_topk.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
template <int BlockThreads, int ItemsPerThread, BlockLoadAlgorithm LoadAlgorithm, BlockStoreAlgorithm StoreAlgorithm>
struct agent_batched_topk_worker_per_segment_policy
{
  /// Threads per thread block
  static constexpr int block_threads = BlockThreads;

  /// Items per thread (per tile of input)
  static constexpr int items_per_thread = ItemsPerThread;

  static constexpr BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
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
struct agent_batched_topk_worker_per_segment
{
  // -------------------------------------------------------------------------
  // Types and Constants
  // -------------------------------------------------------------------------
  // Derive inner types from Iterator of Iterators
  using key_it_t   = it_value_t<KeyInputItItT>;
  using value_it_t = it_value_t<ValueInputItItT>;

  using key_t   = it_value_t<key_it_t>;
  using value_t = it_value_t<value_it_t>;

  static constexpr int block_threads    = ActivePolicyT::block_threads;
  static constexpr int items_per_thread = ActivePolicyT::items_per_thread;
  static constexpr int tile_size        = block_threads * items_per_thread;

  // Check if we are dealing with keys-only or key-value pairs
  static constexpr bool is_keys_only = ::cuda::std::is_same_v<value_t, cub::NullType>;

  // -------------------------------------------------------------------------
  // Primitive Types
  // -------------------------------------------------------------------------
  using block_load_keys_t = BlockLoad<key_t, block_threads, items_per_thread, ActivePolicyT::load_algorithm>;
  using block_load_vals_t = BlockLoad<value_t, block_threads, items_per_thread, ActivePolicyT::load_algorithm>;

  using block_topk_t = block_topk<key_t, block_threads, items_per_thread, value_t>;

  // TODO (elstehle): Specialize for the case that we statically know k and we can skip passing num_valid_items to
  // Store()
  using block_store_keys_t = BlockStore<key_t, block_threads, items_per_thread, ActivePolicyT::store_algorithm>;
  using block_store_vals_t = BlockStore<value_t, block_threads, items_per_thread, ActivePolicyT::store_algorithm>;

  // -------------------------------------------------------------------------
  // Shared Memory Storage
  // -------------------------------------------------------------------------
  struct TempStorage_
  {
    union
    {
      typename block_load_keys_t::TempStorage load_keys;
      typename block_load_vals_t::TempStorage load_vals;
      typename block_topk_t::TempStorage topk;
      typename block_store_keys_t::TempStorage store_keys;
      typename block_store_vals_t::TempStorage store_vals;
    };
  };

  using TempStorage = Uninitialized<TempStorage_>;

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  TempStorage_& temp_storage;
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
  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_worker_per_segment(
    TempStorage& temp_storage,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
      : temp_storage(temp_storage.Alias())
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
      block_topk_t(temp_storage.topk).max_keys(keys, k);
    }
    else
    {
      block_topk_t(temp_storage.topk).min_keys(keys, k);
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
      block_topk_t(temp_storage.topk).max_pairs(keys, values, k);
    }
    else
    {
      block_topk_t(temp_storage.topk).min_pairs(keys, values, k);
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
    if constexpr (params::has_single_static_value_v<SegmentSizeParameterT>
                  && params::static_min_value_v<SegmentSizeParameterT> == tile_size)
    {
      // No padding needed
      block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys);
    }
    else
    {
      // Potentially partial final load with padding
      // TODO (elstehle): explore whether a runtime check for segment_size == tile_size improves performance
      block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys, segment_size, padding_key);
    }

    // Load Values (if applicable)
    [[maybe_unused]] value_t thread_values[items_per_thread];

    if constexpr (!is_keys_only)
    {
      __syncthreads();
      auto block_vals_in = d_value_segments_it[segment_id];

      if constexpr (params::has_single_static_value_v<SegmentSizeParameterT>
                    && params::static_min_value_v<SegmentSizeParameterT> == tile_size)
      {
        // No padding needed
        block_load_vals_t(temp_storage.load_vals).Load(block_vals_in, thread_values);
      }
      else
      {
        // Potentially partial final load with padding
        // TODO (elstehle): explore whether a runtime check for segment_size == tile_size improves performance
        block_load_vals_t(temp_storage.load_vals).Load(block_vals_in, thread_values, segment_size);
      }
    }

    __syncthreads();

    // Perform Block Top-K
    if constexpr (!is_keys_only)
    {
      // Pass both keys and values
      const bool is_successful_dispatch = detail::params::dispatch_discrete(
        select_directions, segment_id, [this, &thread_keys, &thread_values, k](auto direction_tag) {
          select_topk_pairs(thread_keys, thread_values, k, direction_tag);
        });
      _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
    }
    else
    {
      const bool is_successful_dispatch =
        detail::params::dispatch_discrete(select_directions, segment_id, [this, &thread_keys, k](auto direction_tag) {
          select_topk_keys(thread_keys, k, direction_tag);
        });
      _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
    }

    __syncthreads();

    auto block_keys_out = d_key_segments_out_it[segment_id];

    block_store_keys_t(temp_storage.store_keys)
      .Store(block_keys_out,
             thread_keys,
             k // Only store K items
      );

    if constexpr (!is_keys_only)
    {
      __syncthreads();
      auto block_vals_out = d_value_segments_out_it[segment_id];

      block_store_vals_t(temp_storage.store_vals).Store(block_vals_out, thread_values, k);
    }
  }
};
} // namespace detail::batched_topk
CUB_NAMESPACE_END
