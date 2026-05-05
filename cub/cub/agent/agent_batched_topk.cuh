// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Atomic counters used by the small-segment kernel to (a) enqueue large segments into the large-segment work queue
// and (b) elect the last block to run the epilogue scan over the queued tile counts. `alignas(128)` isolates each
// counter on its own cache line for performance.
template <class NumSegmentsT>
struct batched_topk_counters
{
  // Force unsigned integer type for segment count.
  using segment_count_t = detail::choose_offset_t<NumSegmentsT>;
  // Number of segments enqueued in the large-segment work queue. Atomically incremented (by 1) by the first thread
  // of each block that decides its segment is large.
  alignas(128) segment_count_t large_segments_count;

  // Block retirement counter. Each block atomically increments by 1 when it has finished processing its segment, and
  // the block that observes `gridDim.x - 1` runs the epilogue on the queued large segments tile counts.
  // Assumption: Future support for more than 2^31 - 1 segments will use multiple launches of a slightly modified
  // small-segment kernel instead of additional grid dimensions. Therefore each grid will handle a maximum of 2^31 - 1
  // segments per launch. The counter would not even have to be reset to 0 after each launch if we cleverly make use of
  // its modulo arithmetic.
  alignas(128) unsigned retirement_count;
};

template <typename PolicyGetter, // TODO(bgruber): pass worker_policy as NTTP in C++20
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT>
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

  using segment_size_val_t = typename SegmentSizeParameterT::value_type;
  using num_segments_val_t = typename NumSegmentsParameterT::value_type;
  using counters_t         = batched_topk_counters<num_segments_val_t>;

  static constexpr auto policy                 = PolicyGetter{}();
  static constexpr worker_policy active_policy = policy.worker_per_segment_policy;

  // For block-topk (and keys/values load/store):
  static constexpr int block_threads    = active_policy.block_threads;
  static constexpr int items_per_thread = active_policy.items_per_thread;
  static constexpr int tile_size        = block_threads * items_per_thread;

  // For block-scan (and offsets load/store):
  static constexpr int epilogue_items_per_thread = active_policy.epilogue.items_per_thread;
  static constexpr int epilogue_tile_size        = block_threads * epilogue_items_per_thread;

  // Number used for preprocessing segment-size data, not for tuning => should not affect performance of this agent.
  static constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;
  static constexpr int multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.block_threads * multi_worker_per_segment_policy.items_per_thread;

  // Check if there could be large segments present
  static constexpr bool only_small_segments = params::static_max_value_v<SegmentSizeParameterT> <= tile_size;

  // Check if we are dealing with keys-only or key-value pairs
  static constexpr bool is_keys_only = ::cuda::std::is_same_v<value_t, cub::NullType>;

  // -------------------------------------------------------------------------
  // Primitive Types
  // -------------------------------------------------------------------------
  using block_load_keys_t = BlockLoad<key_t, block_threads, items_per_thread, active_policy.load_algorithm>;
  using block_load_vals_t = BlockLoad<value_t, block_threads, items_per_thread, active_policy.load_algorithm>;

  using block_topk_t = block_topk<key_t, block_threads, items_per_thread, value_t>;

  // TODO (elstehle): Specialize for the case that we statically know k and we can skip passing num_valid_items to
  // Store()
  using block_store_keys_t = BlockStore<key_t, block_threads, items_per_thread, active_policy.store_algorithm>;
  using block_store_vals_t = BlockStore<value_t, block_threads, items_per_thread, active_policy.store_algorithm>;

  using block_load_epilogue_t =
    BlockLoad<segment_size_val_t, block_threads, epilogue_items_per_thread, active_policy.epilogue.load_algorithm>;
  using block_scan_epilogue_t = BlockScan<int, block_threads, active_policy.epilogue.scan_algorithm>;
  using block_store_epilogue_t =
    BlockStore<segment_size_val_t, block_threads, epilogue_items_per_thread, active_policy.epilogue.store_algorithm>;

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
      typename block_load_epilogue_t::TempStorage load_epilogue;
      typename block_scan_epilogue_t::TempStorage scan_epilogue;
      typename block_store_epilogue_t::TempStorage store_epilogue;
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
  counters_t* d_counters;
  num_segments_val_t* d_large_segments_ids;
  LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_worker_per_segment(
    TempStorage& temp_storage,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments,
    counters_t* d_counters,
    num_segments_val_t* d_large_segments_ids,
    LargeSegmentTileOffsetT* d_large_segments_tile_offsets)
      : temp_storage(temp_storage.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , select_directions(select_directions)
      , num_segments(num_segments)
      , d_counters(d_counters)
      , d_large_segments_ids(d_large_segments_ids)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
  {}

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
    // Identify Segment
    const int segment_id = static_cast<int>(blockIdx.x);

    // Boundary check
    // TODO (elstehle): consider skipping boundary check if we can safely assume the right grid dimensions
    if (segment_id >= num_segments.get_param(0))
    {
      return;
    }

    constexpr bool is_full_tile = params::has_single_static_value_v<SegmentSizeParameterT>
                               && params::static_min_value_v<SegmentSizeParameterT> == tile_size;

    // Resolve Segment Parameters
    const auto segment_size = segment_sizes.get_param(segment_id);
    if (!only_small_segments && segment_size > tile_size)
    {
      // Enqueue large segment
      if (threadIdx.x == 0u)
      {
        // Add to large segment queue
        const auto large_segment_queue_idx            = atomicAdd(&d_counters->large_segments_count, 1ull);
        d_large_segments_ids[large_segment_queue_idx] = static_cast<num_segments_val_t>(segment_id);
        d_large_segments_tile_offsets[large_segment_queue_idx] =
          static_cast<LargeSegmentTileOffsetT>(::cuda::ceil_div(segment_size, multi_worker_per_segment_tile_size));
      }
    }
    else
    {
      // Process small segment
      const auto k         = (::cuda::std::min) (k_param.get_param(segment_id),
                                         static_cast<decltype(k_param.get_param(segment_id))>(segment_size));
      const auto direction = select_directions.get_param(segment_id);

      // Determine padding key based on direction
      const key_t padding_key =
        (direction == detail::topk::select::max)
          ? ::cuda::std::numeric_limits<key_t>::lowest()
          : ::cuda::std::numeric_limits<key_t>::max();

      // Dereference iterator-of-iterators to get the segment specific iterator
      auto block_keys_in = d_key_segments_it[segment_id];

      // Load Keys
      key_t thread_keys[items_per_thread];
      if constexpr (is_full_tile)
      {
        // No padding needed
        block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys);
      }
      else
      {
        // Potentially partial final load with padding
        // TODO (elstehle): explore whether a runtime check for segment_size == tile_size improves performance
        block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys, segment_size);
      }

      // Load Values (if applicable)
      [[maybe_unused]] value_t thread_values[items_per_thread];

      if constexpr (!is_keys_only)
      {
        __syncthreads();
        auto block_vals_in = d_value_segments_it[segment_id];

        if constexpr (is_full_tile)
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
      if constexpr (is_keys_only)
      {
        const bool is_successful_dispatch = cub::detail::params::dispatch_discrete(
          select_directions, segment_id, [this, &thread_keys, k, segment_size](auto direction_tag) {
            if constexpr (decltype(direction_tag)::value == detail::topk::select::max)
            {
              block_topk_t(temp_storage.topk).template max_keys<is_full_tile>(thread_keys, k, segment_size);
            }
            else
            {
              block_topk_t(temp_storage.topk).template min_keys<is_full_tile>(thread_keys, k, segment_size);
            }
          });
        _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
      }
      else
      {
        // Pass both keys and values
        const bool is_successful_dispatch = cub::detail::params::dispatch_discrete(
          select_directions, segment_id, [this, &thread_keys, &thread_values, k, segment_size](auto direction_tag) {
            if constexpr (decltype(direction_tag)::value == detail::topk::select::max)
            {
              block_topk_t(temp_storage.topk)
                .template max_pairs<is_full_tile>(thread_keys, thread_values, k, segment_size);
            }
            else
            {
              block_topk_t(temp_storage.topk)
                .template min_pairs<is_full_tile>(thread_keys, thread_values, k, segment_size);
            }
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

    // Epilogue: Scan queued large segment sizes (in tiles not elements) for load balancing search in the large segment
    // agent
    if constexpr (!only_small_segments)
    {
      // Determine last block trying to retire.
      bool is_last_block = false;
      if (threadIdx.x == 0u)
      {
        __threadfence();
        const auto retirement_count = atomicAdd(&d_counters->retirement_count, 1u);
        is_last_block               = retirement_count == (gridDim.x - 1u);
      }
      // This sync also makes sure that the shared memory can be reused.
      is_last_block = static_cast<bool>(__syncthreads_or(static_cast<int>(is_last_block)));
      if (!is_last_block)
      {
        return;
      }
      const auto num_large_segments = d_counters->large_segments_count;
      // For tracking the running total across tiles (loop iterations).
      // Caution: The functor is only invoked by the first warp in the block, and the value returned by lane 0 in that
      // warp is used as the initial value.
      const auto prefix_callback_op =
        [running_total = segment_size_val_t{0}](segment_size_val_t block_aggregate) mutable {
          auto old_running_total = running_total;
          running_total += block_aggregate;
          return old_running_total;
        };
      _CCCL_PRAGMA_NOUNROLL()
      for (int large_segment_offset = 0; large_segment_offset < num_large_segments;
           large_segment_offset += epilogue_tile_size)
      {
        segment_size_val_t segment_tile_offsets[epilogue_items_per_thread];
        block_load_epilogue_t(temp_storage.load_epilogue)
          .Load(d_large_segments_tile_offsets + large_segment_offset,
                segment_tile_offsets,
                num_large_segments - large_segment_offset,
                0);
        __syncthreads();
        block_scan_epilogue_t(temp_storage.scan_epilogue)
          .ExclusiveSum(segment_tile_offsets, segment_tile_offsets, prefix_callback_op);
        __syncthreads();
        block_store_epilogue_t(temp_storage.store_epilogue)
          .Store(d_large_segments_tile_offsets + large_segment_offset,
                 segment_tile_offsets,
                 num_large_segments - large_segment_offset);
        __syncthreads();
      }
    }
  }
};
} // namespace detail::batched_topk
CUB_NAMESPACE_END
