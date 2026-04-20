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
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
template <typename PolicyGetter, // TODO(bgruber): pass worker_policy as NTTP in C++20
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

  using segment_size_val_t     = typename SegmentSizeParameterT::value_type;
  using k_val_t                = typename KParameterT::value_type;
  using select_direction_val_t = typename SelectDirectionParameterT::value_type;

  static constexpr worker_policy active_policy = PolicyGetter{}();

  static constexpr int block_threads    = active_policy.block_threads;
  static constexpr int items_per_thread = active_policy.items_per_thread;
  static constexpr int tile_size        = block_threads * items_per_thread;
  static constexpr int epilogue_items_per_thread = active_policy.epilogue_items_per_thread;
  static constexpr int epilogue_tile_size        = block_threads * epilogue_items_per_thread;

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
    BlockLoad<segment_size_val_t, block_threads, epilogue_items_per_thread, active_policy.epilogue_load_algorithm>;
  using block_scan_epilogue_t = BlockScan<int, block_threads, active_policy.epilogue_scan_algorithm>;
  using block_store_epilogue_t =
    BlockStore<segment_size_val_t, block_threads, epilogue_items_per_thread, active_policy.epilogue_store_algorithm>;

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
  // Currently we use int for segment_id, so int has be enough for these atomic counters.
  int* d_retirement_count;
  int* d_large_segment_queue;
  key_it_t* d_key_large_segments_it;
  key_it_t* d_key_large_segments_out_it;
  value_it_t* d_value_large_segments_it;
  value_it_t* d_value_large_segments_out_it;
  [[maybe_unused]] segment_size_val_t* d_large_segments_sizes;
  [[maybe_unused]] k_val_t* d_large_segments_k;
  [[maybe_unused]] select_direction_val_t* d_large_segments_select_directions;
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
    int* d_retirement_count,
    int* d_large_segment_queue,
    key_it_t* d_key_large_segments_it,
    key_it_t* d_key_large_segments_out_it,
    value_it_t* d_value_large_segments_it,
    value_it_t* d_value_large_segments_out_it,
    segment_size_val_t* d_large_segments_sizes,
    k_val_t* d_large_segments_k,
    select_direction_val_t* d_large_segments_select_directions)
      : temp_storage(temp_storage.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , select_directions(select_directions)
      , num_segments(num_segments)
      , d_retirement_count(d_retirement_count)
      , d_large_segment_queue(d_large_segment_queue)
      , d_key_large_segments_it(d_key_large_segments_it)
      , d_key_large_segments_out_it(d_key_large_segments_out_it)
      , d_value_large_segments_it(d_value_large_segments_it)
      , d_value_large_segments_out_it(d_value_large_segments_out_it)
      , d_large_segments_sizes(d_large_segments_sizes)
      , d_large_segments_k(d_large_segments_k)
      , d_large_segments_select_directions(d_large_segments_select_directions)
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
    if (params::static_min_value_v<SegmentSizeParameterT> > tile_size
        || (params::static_max_value_v<SegmentSizeParameterT> > tile_size && segment_size > tile_size))
    {
      // Enqueue large segment
      if (threadIdx.x == 0u)
      {
        // Add to large segment queue
        const int large_segment_queue_index = atomicAdd(d_large_segment_queue, 1);
        // TODO (pauleonix): Should we write out the segment_id instead of all these parameters?
        if constexpr (params::is_per_segment_param_v<SegmentSizeParameterT>)
        {
          d_large_segments_sizes[large_segment_queue_index] = segment_size;
        }
        if constexpr (params::is_per_segment_param_v<KParameterT>)
        {
          d_large_segments_k[large_segment_queue_index] = k_param.get_param(segment_id);
        }
        if constexpr (params::is_per_segment_param_v<SelectDirectionParameterT>)
        {
          d_large_segments_select_directions[large_segment_queue_index] = select_directions.get_param(segment_id);
        }
        d_key_large_segments_it[large_segment_queue_index]     = d_key_segments_it[segment_id];
        d_key_large_segments_out_it[large_segment_queue_index] = d_key_segments_out_it[segment_id];
        if constexpr (!is_keys_only)
        {
          d_value_large_segments_it[large_segment_queue_index]     = d_value_segments_it[segment_id];
          d_value_large_segments_out_it[large_segment_queue_index] = d_value_segments_out_it[segment_id];
        }
      }
    }
    else if constexpr (params::static_min_value_v<SegmentSizeParameterT> <= tile_size)
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

    bool is_last_block = false;
    if (threadIdx.x == 0u)
    {
      __threadfence();
      const int retirement_count = atomicAdd(d_retirement_count, 1);
      is_last_block              = retirement_count == static_cast<int>(gridDim.x) - 1;
    }
    // This sync also makes sure that the shared memory can be reused.
    is_last_block = static_cast<bool>(__syncthreads_or(static_cast<int>(is_last_block)));
    if (!is_last_block)
    {
      return;
    }

    // Epilogue: Scan large segment sizes for load balancing search in the next kernel launch
    const auto num_large_segments = *d_large_segment_queue;
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
      segment_size_val_t segment_sizes[epilogue_items_per_thread];
      block_load_epilogue_t(temp_storage.load_epilogue)
        .Load(
          d_large_segments_sizes + large_segment_offset, segment_sizes, num_large_segments - large_segment_offset, 0);
      __syncthreads();
      block_scan_epilogue_t(temp_storage.scan_epilogue).ExclusiveSum(segment_sizes, segment_sizes, prefix_callback_op);
      __syncthreads();
      block_store_epilogue_t(temp_storage.store_epilogue)
        .Store(d_large_segments_sizes + large_segment_offset, segment_sizes, num_large_segments - large_segment_offset);
      __syncthreads();
    }
  }
};
} // namespace detail::batched_topk
CUB_NAMESPACE_END
