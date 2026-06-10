// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * The cub::BlockHistogramSort class provides sorting-based methods for constructing block-wide
 * histograms from data samples partitioned across a CUDA thread block.
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

#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/util_ptx.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief The BlockHistogramSort class provides sorting-based methods for constructing block-wide
 *        histograms from data samples partitioned across a CUDA thread block.
 *
 * @tparam T
 *   Sample type
 *
 * @tparam BlockDimX
 *   The thread block length in threads along the X dimension
 *
 * @tparam ItemsPerThread
 *   The number of samples per thread
 *
 * @tparam Bins
 *   The number of bins into which histogram samples may fall
 *
 * @tparam BlockDimY
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BlockDimZ
 *   The thread block length in threads along the Z dimension
 */
template <typename T, int BlockDimX, int ItemsPerThread, int Bins, int BlockDimY, int BlockDimZ>
struct BlockHistogramSort
{
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BlockDimX * BlockDimY * BlockDimZ;

  // Parameterize BlockRadixSort type for our thread block
  using BlockRadixSortT =
    BlockRadixSort<T,
                   BlockDimX,
                   ItemsPerThread,
                   NullType,
                   4,
                   true,
                   BLOCK_SCAN_WARP_SCANS,
                   cudaSharedMemBankSizeFourByte,
                   BlockDimY,
                   BlockDimZ>;

  // Parameterize BlockDiscontinuity type for our thread block
  using BlockDiscontinuityT = BlockDiscontinuity<T, BlockDimX, BlockDimY, BlockDimZ>;

  /// Shared memory
  union _TempStorage
  {
    // Storage for sorting bin values
    typename BlockRadixSortT::TempStorage sort;

    struct Discontinuities
    {
      // Storage for detecting discontinuities in the tile of sorted bin values
      typename BlockDiscontinuityT::TempStorage flag;

      // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
      unsigned int run_begin[Bins];
      unsigned int run_end[Bins];
    } discontinuities;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread fields
  _TempStorage& temp_storage;
  unsigned int linear_tid;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramSort(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  // Discontinuity functor
  struct DiscontinuityOp
  {
    // Reference to temp_storage
    _TempStorage& temp_storage;

    // Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE DiscontinuityOp(_TempStorage& temp_storage)
        : temp_storage(temp_storage)
    {}

    // Discontinuity predicate
    _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const T& a, const T& b, int b_index)
    {
      if (a != b)
      {
        // Note the begin/end offsets in shared storage
        temp_storage.discontinuities.run_begin[b] = b_index;
        temp_storage.discontinuities.run_end[a]   = b_index;

        return true;
      }
      else
      {
        return false;
      }
    }
  };

  /**
   * @brief Composite data onto an existing histogram
   *
   * @param[in] items
   *   Calling thread's input values to histogram
   *
   * @param[out] histogram
   *   Reference to shared/device-accessible memory histogram
   */
  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ItemsPerThread], CounterT histogram[Bins])
  {
    static constexpr int TILE_SIZE = BLOCK_THREADS * ItemsPerThread;

    // Sort bytes in blocked arrangement
    BlockRadixSortT(temp_storage.sort).Sort(items);

    __syncthreads();

    // Initialize the shared memory's run_begin and run_end for each bin
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= Bins; histo_offset += BLOCK_THREADS)
    {
      temp_storage.discontinuities.run_begin[histo_offset + linear_tid] = TILE_SIZE;
      temp_storage.discontinuities.run_end[histo_offset + linear_tid]   = TILE_SIZE;
    }
    // Finish up with guarded initialization if necessary
    if ((Bins % BLOCK_THREADS != 0) && (histo_offset + linear_tid < Bins))
    {
      temp_storage.discontinuities.run_begin[histo_offset + linear_tid] = TILE_SIZE;
      temp_storage.discontinuities.run_end[histo_offset + linear_tid]   = TILE_SIZE;
    }

    __syncthreads();

    int flags[ItemsPerThread]; // unused

    // Compute head flags to demarcate contiguous runs of the same bin in the sorted tile
    DiscontinuityOp flag_op(temp_storage);
    BlockDiscontinuityT(temp_storage.discontinuities.flag).FlagHeads(flags, items, flag_op);

    // Update begin for first item
    if (linear_tid == 0)
    {
      temp_storage.discontinuities.run_begin[items[0]] = 0;
    }

    __syncthreads();

    // Composite into histogram
    histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= Bins; histo_offset += BLOCK_THREADS)
    {
      int thread_offset = histo_offset + linear_tid;
      CounterT count =
        temp_storage.discontinuities.run_end[thread_offset] - temp_storage.discontinuities.run_begin[thread_offset];
      histogram[thread_offset] += count;
    }

    // Finish up with guarded composition if necessary
    if ((Bins % BLOCK_THREADS != 0) && (histo_offset + linear_tid < Bins))
    {
      int thread_offset = histo_offset + linear_tid;
      CounterT count =
        temp_storage.discontinuities.run_end[thread_offset] - temp_storage.discontinuities.run_begin[thread_offset];
      histogram[thread_offset] += count;
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
