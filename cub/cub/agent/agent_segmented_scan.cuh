// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file cub::AgentSegmentedScan implements a stateful abstraction of CUDA thread blocks
 *       for participating in device-wide prefix segmented scan .
 */
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

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentSegmentedScan
 *
 * @tparam NOMINAL_BLOCK_THREADS_4B
 *   Threads per thread block
 *
 * @tparam NOMINAL_ITEMS_PER_THREAD_4B
 *   Items per thread (per tile of input)
 *
 * @tparam ComputeT
 *   Dominant compute type
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _LOAD_MODIFIER
 *   Cache load modifier for reading input elements
 *
 * @tparam _STORE_ALGORITHM
 *   The BlockStore algorithm to use
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 */
template <
  int NOMINAL_BLOCK_THREADS_4B,
  int NOMINAL_ITEMS_PER_THREAD_4B,
  typename ComputeT,
  BlockLoadAlgorithm _LOAD_ALGORITHM,
  CacheLoadModifier _LOAD_MODIFIER,
  BlockStoreAlgorithm _STORE_ALGORITHM,
  BlockScanAlgorithm _SCAN_ALGORITHM,
  typename ScalingType = detail::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>>
struct AgentSegmentedScanPolicy : ScalingType
{
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM   = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail::segmented_scan
{

/**
 * @brief AgentSegmentedScan implements a stateful abstraction of CUDA thread blocks for
 *        participating in device-wide segmented prefix scan.
 * @tparam AgentSegmentedScanPolicyT
 *   Parameterized AgentSegmentedScanPolicyT tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
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
  using InputT = cub::detail::it_value_t<InputIteratorT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentSegmentedScanPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  // Constants
  enum
  {
    // Use cub::NullType means no initial value is provided
    HAS_INIT = !::cuda::std::is_same_v<InitValueT, NullType>,
    // We are relying on either initial value not being `NullType`
    // or the ForceInclusive tag to be true for inclusive scan
    // to get picked up.
    IS_INCLUSIVE     = ForceInclusive || !HAS_INIT,
    BLOCK_THREADS    = AgentSegmentedScanPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = AgentSegmentedScanPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
  };

  // Parametrized BlockLoad type
  using BlockLoadT =
    BlockLoad<AccumT,
              AgentSegmentedScanPolicyT::BLOCK_THREADS,
              AgentSegmentedScanPolicyT::ITEMS_PER_THREAD,
              AgentSegmentedScanPolicyT::LOAD_ALGORITHM>;

  // Parametrized BlockStore type
  using BlockStoreT =
    BlockStore<AccumT,
               AgentSegmentedScanPolicyT::BLOCK_THREADS,
               AgentSegmentedScanPolicyT::ITEMS_PER_THREAD,
               AgentSegmentedScanPolicyT::STORE_ALGORITHM>;

  // Parametrized BlockScan type
  using BlockScanT =
    BlockScan<AccumT, AgentSegmentedScanPolicyT::BLOCK_THREADS, AgentSegmentedScanPolicyT::SCAN_ALGORITHM>;

  union _TempStorage
  {
    // smem needed for tile loading
    typename BlockLoadT::TempStorage load;

    // smem needed for tile storing
    typename BlockStoreT::TempStorage store;

    // smem needed for tile scanning
    typename BlockScanT::TempStorage scan;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread-private fields
  _TempStorage& temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associating scan operator
  InitValueT initial_value; // The initial value element for ScanOpT

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_in
   *   Input data
   *
   * @param d_out
   *   Output data
   *
   * @param scan_op
   *   Binary scan operator
   *
   * @param init_value
   *   Initial value to seed the exclusive scan
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSegmentedScan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  /**
   * @brief Scan one segment of values
   *
   * @param inp_idx_begin
   *   Index of start of the segment in input array
   *
   * @param inp_idx_end
   *   Index of end of the segment in input array
   *
   * @param out_idx_begin
   *   Index of start of the segment's prefix scan result in the output array
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT inp_idx_begin, OffsetT inp_idx_end, OffsetT out_idx_begin)
  {
    AccumT exclusive_prefix{};
    OffsetT n_chunks = cuda::ceil_div(inp_idx_end - inp_idx_begin, TILE_ITEMS);

    AccumT thread_values[ITEMS_PER_THREAD] = {};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
      OffsetT chunk_begin = inp_idx_begin + chunk_id * TILE_ITEMS;
      OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + TILE_ITEMS, inp_idx_end);
      // chunk_size <= TILE_ITEMS, casting to int is safe
      int chunk_size = static_cast<int>(chunk_end - chunk_begin);
      // load elements using BlockLoad
      BlockLoadT(temp_storage.load).Load(d_in + chunk_begin, thread_values, chunk_size);
      __syncthreads();

      // execute BlockScan
      AccumT block_aggregate;
      BlockScanT block_scan_algo(temp_storage.scan);
      if constexpr (IS_INCLUSIVE)
      {
        block_scan_algo.InclusiveScan(thread_values, thread_values, scan_op, block_aggregate);
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
        constexpr auto loop_size = static_cast<int>(ITEMS_PER_THREAD);
        cuda::static_for<loop_size>([&](int i) {
          thread_values[i] = thread_values[i] + exclusive_prefix;
        });
        exclusive_prefix = exclusive_prefix + block_aggregate;
      }

      // write out scan values using BlockStore
      BlockStoreT(temp_storage.store).Store(d_out + out_idx_begin + chunk_id * TILE_ITEMS, thread_values, chunk_size);
      __syncthreads();
    }
  };
};

} // namespace detail::segmented_scan

CUB_NAMESPACE_END
