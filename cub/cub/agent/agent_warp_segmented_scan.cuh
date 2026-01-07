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

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/util_arch.cuh>
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
template <typename ComputeT, int NumSegmentsPerWarp>
using agent_warp_segmented_scan_compute_t =
  ::cuda::std::conditional_t<NumSegmentsPerWarp == 1, ComputeT, ::cuda::std::tuple<bool, ComputeT>>;

template <int Nominal4ByteBlockThreads,
          int Nominal4BytesItemsPerThread,
          typename ComputeT,
          WarpLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          WarpStoreAlgorithm StoreAlgorithm,
          int SegmentsPerWarp  = 1,
          typename ScalingType = detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                                         Nominal4BytesItemsPerThread,
                                                         agent_warp_segmented_scan_compute_t<ComputeT, SegmentsPerWarp>>>
struct agent_warp_segmented_scan_policy_t : ScalingType
{
  static_assert(SegmentsPerWarp > 0, "SegmentsPerWarp template value parameter must be positive");

  static constexpr WarpLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier    = LoadModifier;
  static constexpr WarpStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr int segments_per_warp              = SegmentsPerWarp;
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
  static constexpr bool is_inclusive     = ForceInclusive || !has_init;
  static constexpr int block_threads     = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread  = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items        = detail::warp_threads * items_per_thread;
  static constexpr int segments_per_warp = AgentSegmentedScanPolicyT::segments_per_warp;

  static_assert(0 == block_threads % detail::warp_threads, "Block size must be divisible by warp size");

  static constexpr auto warps_in_block = block_threads / detail::warp_threads;

  using augmented_accum_t = agent_warp_segmented_scan_compute_t<AccumT, segments_per_warp>;

  using warp_load_t  = WarpLoad<AccumT, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using warp_store_t = WarpStore<AccumT, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using warp_scan_t  = WarpScan<augmented_accum_t>;

  union _TempStorage
  {
    typename warp_load_t::TempStorage load[warps_in_block];
    typename warp_store_t::TempStorage store[warps_in_block];
    typename warp_scan_t::TempStorage scan[warps_in_block];
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

  template <int NumSegments = segments_per_warp, class = ::cuda::std::enable_if_t<(NumSegments == 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    const OffsetT segment_items = ::cuda::std::max(inp_idx_end, inp_idx_begin) - inp_idx_begin;
    const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, tile_items);

    AccumT exclusive_prefix{};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      const OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      AccumT thread_values[items_per_thread];
      warp_load_t(temp_storage.load[warp_id]).Load(d_in + chunk_begin, thread_values, chunk_size, AccumT{});
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        scan_first_tile(thread_values, initial_value, scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_values, scan_op, exclusive_prefix);
      }
      __syncthreads();

      const OffsetT out_offset = out_idx_begin + chunk_id * tile_items;
      warp_store_t(temp_storage.store[warp_id]).Store(d_out + out_offset, thread_values, chunk_size);
      __syncthreads();
    }
  };

private:
  template <typename ItemTy, typename InitValueTy, typename ScanOpTy, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& warp_aggregate)
  {
    ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);
    warp_scan_t warp_scan_algo(temp_storage.scan[warp_id]);
    if constexpr (has_init)
    {
      warp_scan_algo.ExclusiveScan(thread_aggregate, thread_aggregate, init_value, scan_op, warp_aggregate);
      warp_aggregate = scan_op(init_value, warp_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      warp_scan_algo.ExclusiveScan(thread_aggregate, thread_aggregate, scan_op, warp_aggregate);
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
    warp_scan_t warp_scan_algo(temp_storage.scan[warp_id]);
    ItemTy init_value       = exclusive_prefix;
    ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);
    ItemTy warp_aggregate;
    warp_scan_algo.ExclusiveScan(thread_aggregate, thread_aggregate, init_value, scan_op, warp_aggregate);
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
