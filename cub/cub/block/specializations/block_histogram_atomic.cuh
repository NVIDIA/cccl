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

#if !_CCCL_CUDA_COMPILER(CLANG)
#  include <cuda/__memory/address_space.h>
#  include <cuda/std/__type_traits/is_floating_point.h>
#endif // !_CCCL_CUDA_COMPILER(CLANG)
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>

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
  template <typename T, typename CounterT, int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(const T (&items)[ItemsPerThread], CounterT histogram[Bins])
  {
    // Use unsigned, unsigned long long for all integer counters
    constexpr bool is_integer_counter =
      ::cuda::std::is_integral_v<CounterT> && (sizeof(CounterT) == 4 || sizeof(CounterT) == 8);
    using unsigned_t       = ::cuda::std::conditional_t<sizeof(CounterT) == 4, unsigned, unsigned long long>;
    using atomic_counter_t = ::cuda::std::conditional_t<is_integer_counter, unsigned_t, CounterT>;
    auto* atomic_histogram = reinterpret_cast<atomic_counter_t*>(histogram);

    // Clang does not lower block-scoped atomics to shared-memory instructions
#if !_CCCL_CUDA_COMPILER(CLANG)
    if constexpr (is_integer_counter || ::cuda::std::is_floating_point_v<CounterT>)
    {
      if (::cuda::device::is_address_from(histogram, ::cuda::device::address_space::shared))
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < ItemsPerThread; ++i)
        {
          atomicAdd_block(atomic_histogram + items[i], 1);
        }
        return;
      }
    }
#endif // !_CCCL_CUDA_COMPILER(CLANG)

    // Global histograms need device scope; other counter types may provide atomicAdd through ADL
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      atomicAdd(atomic_histogram + items[i], 1);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
