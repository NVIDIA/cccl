// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * The cub::BlockHistogramAtomic class provides atomic-based methods for constructing block-wide
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

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief The BlockHistogramAtomic class provides atomic-based methods for constructing block-wide
 *        histograms from data samples partitioned across a CUDA thread block.
 */
template <int Bins>
struct BlockHistogramAtomic
{
  /// Shared memory storage layout type
  struct TempStorage
  {};

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramAtomic(TempStorage& temp_storage) {}

  /**
   * @brief Composite data onto an existing histogram
   *
   * @param[in] items
   *   Calling thread's input values to histogram
   *
   * @param[out] histogram
   *   Reference to shared/device-accessible memory histogram
   */
  template <typename T, typename CounterT, int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT histogram[Bins])
  {
    // Update histogram
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      atomicAdd(histogram + items[i], 1);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
