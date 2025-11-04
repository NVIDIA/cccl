// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//!
//! @file cub::AgentSegmentedScan implements a stateful abstraction of CUDA thread blocks
//!       for participating in device-wide prefix segmented scan .
//!
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
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

//!
//! @brief Parameterizable tuning policy type for AgentSegmentedScan
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
template <
  int Nominal4ByteBlockThreads,
  int Nominal4BytesItemsPerThread,
  typename ComputeT,
  BlockLoadAlgorithm LoadAlgorithm,
  CacheLoadModifier LoadModifier,
  BlockStoreAlgorithm StoreAlgorithm,
  BlockScanAlgorithm ScanAlgorithm,
  typename ScalingType = detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4BytesItemsPerThread, ComputeT>>
struct AgentSegmentedScanPolicy : ScalingType
{
  static constexpr BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier     = LoadModifier;
  static constexpr BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr BlockScanAlgorithm scan_algorithm   = ScanAlgorithm;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

//!
//! @brief AgentSegmentedScan implements a stateful abstraction of CUDA thread blocks for
//!        participating in device-wide segmented prefix scan.
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
struct AgentSegmentedScan
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
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
  static constexpr bool is_inclusive    = ForceInclusive || !has_init;
  static constexpr int block_threads    = AgentSegmentedScanPolicyT::BLOCK_THREADS;
  static constexpr int items_per_thread = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using block_load_t = BlockLoad<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::load_algorithm>;

  using block_store_t = BlockStore<AccumT, block_threads, items_per_thread, AgentSegmentedScanPolicyT::store_algorithm>;

  using block_scan_t = BlockScan<AccumT, block_threads, AgentSegmentedScanPolicyT::scan_algorithm>;

  union _TempStorage
  {
    // smem needed for tile loading
    typename block_load_t::TempStorage load;

    // smem needed for tile storing
    typename block_store_t::TempStorage store;

    // smem needed for tile scanning
    typename block_scan_t::TempStorage scan;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread-private fields
  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associating scan operator
  InitValueT initial_value; // The initial value element for ScanOpT

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  //!
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
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSegmentedScan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  //!
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    AccumT exclusive_prefix{};
    OffsetT n_chunks = ::cuda::ceil_div(inp_idx_end - inp_idx_begin, tile_items);

    AccumT thread_values[items_per_thread] = {};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      OffsetT chunk_begin = inp_idx_begin + chunk_id * tile_items;
      OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, inp_idx_end);
      // chunk_size <= TILE_ITEMS, casting to int is safe
      int chunk_size = static_cast<int>(chunk_end - chunk_begin);
      // load elements using BlockLoad
      block_load_t(temp_storage.load).Load(d_in + chunk_begin, thread_values, chunk_size);
      __syncthreads();

      // execute BlockScan
      AccumT block_aggregate;
      block_scan_t block_scan_algo(temp_storage.scan);
      if constexpr (is_inclusive)
      {
        if constexpr (has_init)
        {
          if (chunk_id == 0)
          {
            block_scan_algo.InclusiveScan(thread_values, thread_values, initial_value, scan_op, block_aggregate);
            block_aggregate = scan_op(initial_value, block_aggregate);
          }
          else
          {
            block_scan_algo.InclusiveScan(thread_values, thread_values, scan_op, block_aggregate);
          }
        }
        else
        {
          block_scan_algo.InclusiveScan(thread_values, thread_values, scan_op, block_aggregate);
        }
      }
      else
      {
        block_scan_algo.ExclusiveScan(thread_values, thread_values, initial_value, scan_op, block_aggregate);
      }
      __syncthreads();

      // update values in registers with exclusive_prefix
      if (chunk_id == 0)
      {
        exclusive_prefix = block_aggregate;
      }
      else
      {
        constexpr auto loop_size = static_cast<int>(items_per_thread);
        cuda::static_for<loop_size>([&](int i) {
          thread_values[i] = scan_op(thread_values[i], exclusive_prefix);
        });
        exclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
      }

      // write out scan values using BlockStore
      block_store_t(temp_storage.store).Store(d_out + out_idx_begin + chunk_id * tile_items, thread_values, chunk_size);
      __syncthreads();
    }
  };
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
