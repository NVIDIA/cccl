// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::detail::segmented_scan::AgentSegmentedScan implements a stateful abstraction of CUDA thread
//! blocks for participating in device-wide prefix segmented scan.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/segmented_scan_multi_segment_helpers.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_reduce.cuh> // ThreadReduce
#include <cub/thread/thread_scan.cuh> // detail::ThreadInclusiveScan
#include <cub/util_arch.cuh> // detail::MemBoundScaling
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename ComputeT, int MaxSegmentsPerWarp>
using agent_warp_segmented_scan_compute_t =
  multi_segment_helpers::agent_segmented_scan_compute_t<ComputeT, MaxSegmentsPerWarp>;

template <
  int Nominal4ByteBlockThreads,
  int Nominal4BytesItemsPerThread,
  typename ComputeT,
  WarpLoadAlgorithm LoadAlgorithm,
  CacheLoadModifier LoadModifier,
  WarpStoreAlgorithm StoreAlgorithm,
  int MaxSegmentsPerWarp = 1,
  typename ScalingType   = detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                                   Nominal4BytesItemsPerThread,
                                                   agent_warp_segmented_scan_compute_t<ComputeT, MaxSegmentsPerWarp>>>
struct agent_warp_segmented_scan_policy_t : ScalingType
{
  static_assert(MaxSegmentsPerWarp > 0, "MaxSegmentsPerWarp template value parameter must be positive");

  static constexpr WarpLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier    = LoadModifier;
  static constexpr WarpStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr int max_segments_per_warp          = MaxSegmentsPerWarp;
};

template <typename AgentSegmentedScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_warp_segmented_scan
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using input_t = cub::detail::it_value_t<InputIteratorT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using wrapped_input_iterator_t =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentSegmentedScanPolicyT::load_modifier, input_t, OffsetT>,
                     InputIteratorT>;

  // Constants

  // Use cub::NullType means no initial value is provided
  static constexpr bool has_init = !::cuda::std::is_same_v<InitValueT, NullType>;
  // We are relying on either initial value not being `NullType`
  // or the ForceInclusive tag to be true for inclusive scan
  // to get picked up.
  static constexpr bool is_inclusive         = ForceInclusive || !has_init;
  static constexpr int block_threads         = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread      = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items            = detail::warp_threads * items_per_thread;
  static constexpr int max_segments_per_warp = AgentSegmentedScanPolicyT::max_segments_per_warp;

  static_assert(0 == block_threads % detail::warp_threads, "Block size must be divisible by warp size");

  static constexpr auto warps_in_block = block_threads / detail::warp_threads;

  using augmented_accum_t = agent_warp_segmented_scan_compute_t<AccumT, max_segments_per_warp>;

  using warp_load_t  = WarpLoad<augmented_accum_t, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using warp_store_t = WarpStore<augmented_accum_t, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using warp_scan_t  = WarpScan<augmented_accum_t>;
  using warp_scan_offsets_t = WarpScan<OffsetT>;

  struct _TempStorage
  {
    OffsetT logical_segment_offsets[warps_in_block][max_segments_per_warp];
    union AlgorithmsStorage
    {
      typename warp_load_t::TempStorage load[warps_in_block];
      typename warp_store_t::TempStorage store[warps_in_block];
      typename warp_scan_t::TempStorage scan[warps_in_block];
      typename warp_scan_offsets_t::TempStorage offsets_scan[warps_in_block];
    } reused;
  };

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT
  unsigned int warp_id;
  unsigned int lane_id;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_warp_segmented_scan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
      , warp_id(threadIdx.x >> cub::detail::log2_warp_threads)
      , lane_id(threadIdx.x % cub::detail::warp_threads)
  {}

  template <int NumSegments = max_segments_per_warp, class = ::cuda::std::enable_if_t<(NumSegments == 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    const OffsetT segment_items = ::cuda::std::max(inp_idx_end, inp_idx_begin) - inp_idx_begin;
    const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, tile_items);

    AccumT exclusive_prefix{};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      AccumT thread_values[items_per_thread];
      if (chunk_size == tile_items)
      {
        warp_load_t(temp_storage.reused.load[warp_id]).Load(d_in + chunk_begin, thread_values);
      }
      else
      {
        constexpr AccumT oob_default{};
        warp_load_t(temp_storage.reused.load[warp_id]).Load(d_in + chunk_begin, thread_values, chunk_size, oob_default);
      }
      __syncwarp();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        scan_first_tile(thread_values, initial_value, scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_values, scan_op, exclusive_prefix);
      }
      __syncwarp();

      const OffsetT out_offset = out_idx_begin + chunk_id * tile_items;
      if (chunk_size == tile_items)
      {
        warp_store_t(temp_storage.reused.store[warp_id]).Store(d_out + out_offset, thread_values);
      }
      else
      {
        warp_store_t(temp_storage.reused.store[warp_id]).Store(d_out + out_offset, thread_values, chunk_size);
      }
      if (++chunk_id < n_chunks)
      {
        __syncwarp();
      }
    }
  };

  //! @brief Scan dynamically given number of segment of values
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = max_segments_per_warp,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");

    _CCCL_ASSERT(n_segments > 0, "Number of segments should be greater than zero");
    _CCCL_ASSERT(n_segments <= max_segments_per_warp,
                 "Number of segments should not exceed statically provisioned storage");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block
    {
      n_segments                        = ::cuda::std::min(n_segments, static_cast<int>(NumSegments));
      constexpr unsigned worker_threads = cub::detail::warp_threads;
      unsigned n_chunks                 = ::cuda::ceil_div<unsigned>(n_segments, worker_threads);
      OffsetT exclusive_prefix          = 0;
      using plus_t                      = ::cuda::std::plus<>;
      const plus_t offsets_scan_op{};
      worker_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

      warp_scan_offsets_t offset_scan_algo(temp_storage.reused.offsets_scan[warp_id]);

      const unsigned lane_id = (threadIdx.x % worker_threads);
      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * worker_threads + lane_id;

        // TODO: use WarpLoad to load?
        const OffsetT input_segment_begin = (work_id < n_segments) ? inp_idx_begin_it[work_id] : 0;
        const OffsetT input_segment_end   = (work_id < n_segments) ? inp_idx_end_it[work_id] : 0;
        const OffsetT segment_size        = input_segment_end - input_segment_begin;

        OffsetT prefix;
        OffsetT warp_aggregate;
        offset_scan_algo.InclusiveSum(segment_size, prefix, warp_aggregate);
        __syncwarp();
        OffsetT warp_prefix = prefix_callback_op(warp_aggregate);

        if (work_id < n_segments)
        {
          temp_storage.logical_segment_offsets[warp_id][work_id] = warp_prefix + prefix;
        }
      }
    }

    __syncthreads();

    ::cuda::std::span<OffsetT> cum_sizes{
      temp_storage.logical_segment_offsets[warp_id], static_cast<::cuda::std::size_t>(n_segments)};
    const OffsetT items_per_block = cum_sizes[n_segments - 1];
    const OffsetT n_chunks        = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_scan_op_t = multi_segment_helpers::schwarz_scan_op<AccumT, bool, ScanOpT>;
    using augmented_init_value_t =
      ::cuda::std::conditional_t<has_init, augmented_accum_t, multi_segment_helpers::augmented_value_t<InitValueT, bool>>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};

    using multi_segment_helpers::multi_segmented_iterator;
    using multi_segment_helpers::packer;
    using multi_segment_helpers::packer_iv;
    using multi_segment_helpers::projector;
    using multi_segment_helpers::projector_iv;

    augmented_accum_t thread_flag_values[items_per_thread];
    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      {
        constexpr auto oob_default = multi_segment_helpers::make_value_flag(AccumT{}, false);
        constexpr projector<AccumT, bool> projection_op{};

        warp_load_t loader(temp_storage.reused.load[warp_id]);
        if constexpr (has_init)
        {
          // If initial value is provided, it should be applied to segment's head value
          const packer_iv<AccumT, bool, ScanOpT> packer_op{static_cast<AccumT>(initial_value), scan_op};
          multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it, packer_op, projection_op};

          if (chunk_size == tile_items)
          {
            loader.Load(it_in, thread_flag_values);
          }
          else
          {
            loader.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
        else
        {
          constexpr packer<AccumT, bool> packer_op{};
          multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it, packer_op, projection_op};

          if (chunk_size == tile_items)
          {
            loader.Load(it_in, thread_flag_values);
          }
          else
          {
            loader.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
      }
      __syncwarp();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value = multi_segment_helpers::make_value_flag(initial_value, false);
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, exclusive_prefix);
      }
      __syncwarp();

      // store prefix-scan values, discarding head flags
      {
        constexpr packer<AccumT, bool> packer_op{};
        const OffsetT out_offset = chunk_id * tile_items;

        warp_store_t storer(temp_storage.reused.store[warp_id]);
        if constexpr (is_inclusive)
        {
          constexpr projector<AccumT, bool> projector_op{};
          multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it, packer_op, projector_op};

          if (chunk_size == tile_items)
          {
            storer.Store(it_out, thread_flag_values);
          }
          else
          {
            storer.Store(it_out, thread_flag_values, chunk_size);
          }
        }
        else
        {
          const projector_iv<AccumT, bool> projector_op{static_cast<AccumT>(initial_value)};
          multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it, packer_op, projector_op};
          if (chunk_size == tile_items)
          {
            storer.Store(it_out, thread_flag_values);
          }
          else
          {
            storer.Store(it_out, thread_flag_values, chunk_size);
          }
        }
      }

      if (++chunk_id < n_chunks)
      {
        __syncwarp();
      }
    }
  }

private:
  template <typename ItemTy,
            typename InitValueTy,
            typename ScanOpTy,
            bool IsInclusive = is_inclusive,
            bool HasInit     = has_init>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& warp_aggregate)
  {
    // TODO: specialize for items_per_thread == 1
    ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);
    warp_scan_t scanner(temp_storage.reused.scan[warp_id]);
    if constexpr (HasInit)
    {
      scanner.ExclusiveScan(thread_aggregate, thread_aggregate, init_value, scan_op, warp_aggregate);
      warp_aggregate = scan_op(init_value, warp_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      scanner.ExclusiveScan(thread_aggregate, thread_aggregate, scan_op, warp_aggregate);
    }
    if constexpr (IsInclusive)
    {
      detail::ThreadScanInclusive(items, items, scan_op, thread_aggregate, has_init || (lane_id != 0));
    }
    else
    {
      detail::ThreadScanExclusive(items, items, scan_op, thread_aggregate, has_init || (lane_id != 0));
    }
  }

  template <typename ItemTy, typename ScanOpTy, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ItemTy (&items)[items_per_thread], ScanOpTy scan_op, ItemTy& exclusive_prefix)
  {
    // TODO: specialize for items_per_thread == 1
    warp_scan_t scanner(temp_storage.reused.scan[warp_id]);
    const ItemTy& init_value = exclusive_prefix;
    ItemTy thread_aggregate  = cub::ThreadReduce(items, scan_op);
    ItemTy warp_aggregate;

    scanner.ExclusiveScan(thread_aggregate, thread_aggregate, init_value, scan_op, warp_aggregate);

    if constexpr (IsInclusive)
    {
      detail::ThreadScanInclusive(items, items, scan_op, thread_aggregate);
    }
    else
    {
      detail::ThreadScanExclusive(items, items, scan_op, thread_aggregate);
    }
    exclusive_prefix = scan_op(exclusive_prefix, warp_aggregate);
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
