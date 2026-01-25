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

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/segmented_scan_multi_segment_helpers.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_arch.cuh> // MemBoundScaling

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

using multi_segment_helpers::augmented_value_t;

template <typename ComputeT, int NumSegmentsPerBlock>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<NumSegmentsPerBlock == 1, ComputeT, augmented_value_t<ComputeT, bool>>;

//! @brief Parameterizable tuning policy type for agent_segmented_scan
//!
//! @tparam Nominal4ByteBlockThreads
//!   Threads per thread block
//!
//! @tparam Nominal4BytesItemsPerThread
//!   Items per thread (per tile of input)
//!
//! @tparam ComputeT
//!   Dominant compute type
//!
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam LoadModifier
//!   Cache load modifier for reading input elements
//!
//! @tparam StoreAlgorithm
//!   The BlockStore algorithm to use
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
//! @tparam SegmentsPerBlock
//!   The number of segments processed per block
//!
template <int Nominal4ByteBlockThreads,
          int Nominal4BytesItemsPerThread,
          typename ComputeT,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          BlockStoreAlgorithm StoreAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          int MaxSegmentsPerBlock = 1,
          typename ScalingType    = detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                                            Nominal4BytesItemsPerThread,
                                                            agent_segmented_scan_compute_t<ComputeT, MaxSegmentsPerBlock>>>
struct agent_segmented_scan_policy_t : ScalingType
{
  static_assert(MaxSegmentsPerBlock > 0, "MaxSegmentsPerBlock template value parameter must be positive");

  static constexpr BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier     = LoadModifier;
  static constexpr BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr BlockScanAlgorithm scan_algorithm   = ScanAlgorithm;
  static constexpr int max_segments_per_block          = MaxSegmentsPerBlock;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

//! @brief agent_segmented_scan implements a stateful abstraction of CUDA thread blocks for
//!        participating in device-wide segmented prefix scan.
//!
//! @tparam AgentSegmentedScanPolicyT
//!   Parameterized AgentSegmentedScanPolicyT tuning policy type
//!
//! @tparam InputIteratorT
//!   Random-access input iterator type
//!
//! @tparam OutputIteratorT
//!   Random-access output iterator type
//!
//! @tparam OffsetT
//!   Signed integer type for global offsets
//!
//! @tparam ScanOpT
//!   Scan functor type
//!
//! @tparam InitValueT
//!   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
//!
//! @tparam AccumT
//!   The type of intermediate accumulator (according to P2322R6)
//!
template <typename AgentSegmentedScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_segmented_scan
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
  static constexpr bool is_inclusive          = ForceInclusive || !has_init;
  static constexpr int block_threads          = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread       = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items             = block_threads * items_per_thread;
  static constexpr int max_segments_per_block = AgentSegmentedScanPolicyT::max_segments_per_block;

  using augmented_accum_t = agent_segmented_scan_compute_t<AccumT, max_segments_per_block>;

  using block_load_t =
    BlockLoad<augmented_accum_t, block_threads, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;
  using block_store_t =
    BlockStore<augmented_accum_t, block_threads, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;
  using block_scan_t        = BlockScan<augmented_accum_t, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;
  using block_offset_scan_t = BlockScan<OffsetT, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;

  struct _TempStorage
  {
    OffsetT logical_segment_offsets[max_segments_per_block];
    union AlgorithmsStorage
    {
      typename block_load_t::TempStorage load;
      typename block_store_t::TempStorage store;
      typename block_scan_t::TempStorage scan;
      typename block_offset_scan_t::TempStorage offset_scan;
    } reused;
  };

  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  //! @param temp_storage
  //!   Reference to temp_storage
  //!
  //! @param d_in
  //!   Input data
  //!
  //! @param d_out
  //!   Output data
  //!
  //! @param scan_op
  //!   Binary scan operator
  //!
  //! @param init_value
  //!   Initial value to seed the exclusive scan
  //!
  _CCCL_DEVICE _CCCL_FORCEINLINE agent_segmented_scan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  //! @brief Scan one segment of values
  //!
  //! @param inp_idx_begin
  //!   Index of start of the segment in input array
  //!
  //! @param inp_idx_end
  //!  Index of end of the segment in input array
  //!
  //! @param out_idx_begin
  //!  Index of start of the segment's prefix scan result in the output array
  //!
  template <int NumSegments = max_segments_per_block, class = ::cuda::std::enable_if_t<(NumSegments == 1)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_range(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    const OffsetT segment_items = ::cuda::std::max(inp_idx_end, inp_idx_begin) - inp_idx_begin;
    const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, tile_items);

    AccumT exclusive_prefix{};
    block_prefix_callback_t prefix_op{exclusive_prefix, scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      AccumT thread_values[items_per_thread];
      if (chunk_size == tile_items)
      {
        block_load_t(temp_storage.reused.load).Load(d_in + chunk_begin, thread_values);
      }
      else
      {
        block_load_t(temp_storage.reused.load).Load(d_in + chunk_begin, thread_values, chunk_size, AccumT{});
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        scan_first_tile(thread_values, initial_value, scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_values, scan_op, prefix_op);
      }
      __syncthreads();

      if (chunk_size == tile_items)
      {
        block_store_t(temp_storage.reused.store).Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values);
      }
      else
      {
        block_store_t(temp_storage.reused.store)
          .Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values, chunk_size);
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  };

  //! @brief Scan dynamically given number of segment of values
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments = max_segments_per_block,
            class = ::cuda::std::enable_if_t<(NumSegments > 1) && (NumSegments != ::cuda::std::dynamic_extent)>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_ranges(
    InputBeginOffsetIteratorT inp_idx_begin_it,
    InputEndOffsetIteratorT inp_idx_end_it,
    OutputBeginOffsetIteratorT out_idx_begin_it,
    int n_segments)
  {
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<InputEndOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_value_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");

    _CCCL_ASSERT(n_segments > 0, "Number of segments per worker should be positive");
    _CCCL_ASSERT(n_segments <= NumSegments, "Number of segments per worker exceeds statically provisioned storage");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block
    {
      n_segments               = ::cuda::std::min(n_segments, static_cast<int>(NumSegments));
      unsigned n_chunks        = ::cuda::ceil_div<unsigned>(n_segments, block_threads);
      OffsetT exclusive_prefix = 0;
      using plus_t             = ::cuda::std::plus<>;
      const plus_t offsets_scan_op{};
      block_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * block_threads + threadIdx.x;

        // TODO: use BlockLoad to load
        const OffsetT input_segment_begin = (work_id < n_segments) ? inp_idx_begin_it[work_id] : 0;
        const OffsetT input_segment_end   = (work_id < n_segments) ? inp_idx_end_it[work_id] : 0;
        const OffsetT segment_size        = input_segment_end - input_segment_begin;

        OffsetT prefix;
        block_offset_scan_t(temp_storage.reused.offset_scan).InclusiveSum(segment_size, prefix, prefix_callback_op);
        __syncthreads();

        temp_storage.logical_segment_offsets[work_id] = prefix;
      }
    }

    __syncthreads();

    ::cuda::std::span<OffsetT> cum_sizes{
      temp_storage.logical_segment_offsets, static_cast<::cuda::std::size_t>(n_segments)};
    const OffsetT items_per_block = cum_sizes[n_segments - 1];

    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_scan_op_t = multi_segment_helpers::schwarz_scan_op<AccumT, bool, ScanOpT>;
    using augmented_init_value_t =
      ::cuda::std::conditional_t<has_init, augmented_accum_t, augmented_value_t<InitValueT, bool>>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    block_prefix_callback_t prefix_op{exclusive_prefix, augmented_scan_op};

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

        block_load_t load_algo(temp_storage.reused.load);
        if constexpr (has_init)
        {
          const packer_iv<AccumT, bool, ScanOpT> packer_op{static_cast<AccumT>(initial_value), scan_op};
          multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it, packer_op, projection_op};

          if (chunk_size == tile_items)
          {
            load_algo.Load(it_in, thread_flag_values);
          }
          else
          {
            load_algo.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
        else
        {
          constexpr packer<AccumT, bool> packer_op{};
          multi_segmented_iterator it_in{d_in, chunk_begin, cum_sizes, inp_idx_begin_it, packer_op, projection_op};

          if (chunk_size == tile_items)
          {
            load_algo.Load(it_in, thread_flag_values);
          }
          else
          {
            load_algo.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
      }
      __syncthreads();

      if (chunk_id == 0)
      {
        // Initialize exlusive_prefix, referenced from prefix_op
        augmented_init_value_t augmented_init_value = multi_segment_helpers::make_value_flag(initial_value, false);
        scan_first_tile(thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
      }
      else
      {
        scan_later_tile(thread_flag_values, augmented_scan_op, prefix_op);
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        constexpr packer<AccumT, bool> packer_op{};
        const OffsetT out_offset = chunk_id * tile_items;
        block_store_t store_algo(temp_storage.reused.store);
        if constexpr (is_inclusive)
        {
          constexpr projector<AccumT, bool> projector_op{};
          multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it, packer_op, projector_op};

          if (chunk_size == tile_items)
          {
            store_algo.Store(it_out, thread_flag_values);
          }
          else
          {
            store_algo.Store(it_out, thread_flag_values, chunk_size);
          }
        }
        else
        {
          const projector_iv<AccumT, bool> projector_op{static_cast<AccumT>(initial_value)};
          multi_segmented_iterator it_out{d_out, out_offset, cum_sizes, out_idx_begin_it, packer_op, projector_op};
          if (chunk_size == tile_items)
          {
            store_algo.Store(it_out, thread_flag_values);
          }
          else
          {
            store_algo.Store(it_out, thread_flag_values, chunk_size);
          }
        }
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
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
  scan_first_tile(ItemTy (&items)[items_per_thread], InitValueTy init_value, ScanOpTy scan_op, ItemTy& block_aggregate)
  {
    block_scan_t block_scan_algo(temp_storage.reused.scan);
    if constexpr (HasInit)
    {
      if constexpr (IsInclusive)
      {
        block_scan_algo.InclusiveScan(items, items, init_value, scan_op, block_aggregate);
      }
      else
      {
        block_scan_algo.ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
      }
      block_aggregate = scan_op(init_value, block_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      block_scan_algo.InclusiveScan(items, items, scan_op, block_aggregate);
    }
  }

  template <typename ItemTy, typename ScanOpTy, typename PrefixCallback, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ItemTy (&items)[items_per_thread], ScanOpTy scan_op, PrefixCallback& prefix_op)
  {
    block_scan_t block_scan_algo(temp_storage.reused.scan);
    if constexpr (IsInclusive)
    {
      block_scan_algo.InclusiveScan(items, items, scan_op, prefix_op);
    }
    else
    {
      block_scan_algo.ExclusiveScan(items, items, scan_op, prefix_op);
    }
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
