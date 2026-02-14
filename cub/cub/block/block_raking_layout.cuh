// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::BlockRakingLayout provides a conflict-free shared memory layout abstraction for warp-raking
 * across thread block data.
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

#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @rst
//! BlockRakingLayout provides a conflict-free shared memory layout abstraction for 1D raking across thread block data.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! This type facilitates a shared memory usage pattern where a block of CUDA
//! threads places elements into shared memory and then reduces the active
//! parallelism to one "raking" warp of threads for serially aggregating consecutive
//! sequences of shared items.  Padding is inserted to eliminate bank conflicts
//! (for most data types).
//!
//! @endrst
//!
//! @tparam T
//!   The data type to be exchanged.
//!
//! @tparam BlockThreads
//!   The thread block size in threads.
//!
template <typename T, int BlockThreads>
struct BlockRakingLayout
{
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  /// The total number of elements that need to be cooperatively reduced
  static constexpr int SHARED_ELEMENTS = BlockThreads;

  /// Maximum number of warp-synchronous raking threads
  static constexpr int MAX_RAKING_THREADS = ::cuda::std::min(BlockThreads, detail::warp_threads);

  /// Number of raking elements per warp-synchronous raking thread (rounded up)
  static constexpr int SEGMENT_LENGTH = (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS;

  /// Never use a raking thread that will have no valid data (e.g., when BlockThreads is 62 and SEGMENT_LENGTH is 2,
  /// we should only use 31 raking threads)
  static constexpr int RAKING_THREADS = (SHARED_ELEMENTS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;

  /// Whether we will have bank conflicts (technically we should find out if the GCD is > 1)
  static constexpr bool HAS_CONFLICTS = (detail::smem_banks % SEGMENT_LENGTH == 0);

  /// Degree of bank conflicts (e.g., 4-way)
  static constexpr int CONFLICT_DEGREE =
    (HAS_CONFLICTS) ? (MAX_RAKING_THREADS * SEGMENT_LENGTH) / detail::smem_banks : 1;

  /// Pad each segment length with one element if segment length is not relatively prime to warp size and can't be
  /// optimized as a vector load
  static constexpr bool USE_SEGMENT_PADDING = ((SEGMENT_LENGTH & 1) == 0) && (SEGMENT_LENGTH > 2);

  /// Total number of elements in the raking grid
  static constexpr int GRID_ELEMENTS = RAKING_THREADS * (SEGMENT_LENGTH + USE_SEGMENT_PADDING);

  /// Whether or not we need bounds checking during raking (the number of reduction elements is not a multiple of the
  /// number of raking threads)
  static constexpr int UNGUARDED = (SHARED_ELEMENTS % RAKING_THREADS == 0);

  /**
   * @brief Shared memory storage type
   */
  struct __align__(16) _TempStorage
  {
    T buff[BlockRakingLayout::GRID_ELEMENTS];
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  /**
   * @brief Returns the location for the calling thread to place data into the grid
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE T* PlacementPtr(TempStorage& temp_storage, unsigned int linear_tid)
  {
    // Offset for partial
    unsigned int offset = linear_tid;

    // Add in one padding element for every segment
    if (USE_SEGMENT_PADDING > 0)
    {
      offset += offset / SEGMENT_LENGTH;
    }

    // Incorporating a block of padding partials every shared memory segment
    return temp_storage.Alias().buff + offset;
  }

  /**
   * @brief Returns the location for the calling thread to begin sequential raking
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE T* RakingPtr(TempStorage& temp_storage, unsigned int linear_tid)
  {
    return temp_storage.Alias().buff + (linear_tid * (SEGMENT_LENGTH + USE_SEGMENT_PADDING));
  }
};

CUB_NAMESPACE_END
