// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/swap.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename KeyT, typename ValueT, typename CompareOp>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
compare_swap(KeyT& key_lhs, KeyT& key_rhs, ValueT& item_lhs, ValueT& item_rhs, CompareOp compare_op)
{
  if (compare_op(key_rhs, key_lhs))
  {
    using ::cuda::std::swap;
    swap(key_lhs, key_rhs);
    if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
    {
      swap(item_lhs, item_rhs);
    }
  }
}

template <bool Unroll = true, typename KeyT, typename ValueT, typename CompareOp, int ItemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
stable_odd_even_sort(KeyT (&keys)[ItemPerThread], ValueT (&items)[ItemPerThread], CompareOp compare_op)
{
  constexpr int unroll = Unroll ? ItemPerThread : 1;
  _CCCL_PRAGMA_UNROLL(unroll)
  for (int i = 0; i < ItemPerThread; ++i)
  {
    _CCCL_PRAGMA_UNROLL(unroll) // unroll count is higher than loop count, but that's fine
    for (int j = i % 2; j < ItemPerThread - 1; j += 2)
    {
      cub::detail::compare_swap(keys[j], keys[j + 1], items[j], items[j + 1], compare_op);
    } // inner loop
  } // outer loop
}

template <typename KeyT, typename ValueT, typename CompareOp, int ItemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
unstable_pairwise_sort(KeyT (&keys)[ItemPerThread], ValueT (&items)[ItemPerThread], CompareOp compare_op)
{
  constexpr int network_degree = cuda::ceil_ilog2(ItemPerThread);
  constexpr int network_size   = cuda::next_power_of_two(ItemPerThread);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int group_size = 1; group_size < network_size; group_size *= 2)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int group_offset = 0; group_offset < group_size; ++group_offset)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int lhs = group_offset; lhs < ItemPerThread - group_size; lhs += 2 * group_size)
      {
        const int rhs = lhs + group_size;
        cub::detail::compare_swap(keys[lhs], keys[rhs], items[lhs], items[rhs], compare_op);
      }
    }
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int stage = 0; stage < network_degree - 1; ++stage)
  {
    const int group_size = network_size >> (stage + 2); // network_size / 2^(stage + 2)
    const int first_step = (1 << (stage + 1)) - 1; // 2^(stage + 1) - 1

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int step = first_step; step > 0; step /= 2)
    {
      const int stride = group_size * step;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int lhs = 0; lhs < ItemPerThread; ++lhs)
      {
        if ((lhs / group_size) % 2 == 1)
        {
          const int rhs = lhs + stride;
          if (rhs < ItemPerThread)
          {
            cub::detail::compare_swap(keys[lhs], keys[rhs], items[lhs], items[rhs], compare_op);
          }
        }
      }
    }
  }
}
} // namespace detail

/**
 * @brief Sorts data using odd-even sort method
 *
 * The sorting method is stable. Further details can be found in:
 * A. Nico Habermann. Parallel neighbor sort (or the glory of the induction principle). Technical Report AD-759 248,
 * Carnegie Mellon University, 1972.
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type. If `cub::NullType` is used as `ValueT`, only keys are sorted.
 *
 * @tparam CompareOp
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`
 *
 * @tparam ItemPerThread
 *   The number of items per thread
 *
 * @param[in,out] keys
 *   Keys to sort
 *
 * @param[in,out] items
 *   Values to sort
 *
 * @param[in] compare_op
 *   Comparison function object which returns true if the first argument is
 *   ordered before the second
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ItemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StableOddEvenSort(KeyT (&keys)[ItemPerThread], ValueT (&items)[ItemPerThread], CompareOp compare_op)
{
  return cub::detail::stable_odd_even_sort(keys, items, compare_op);
}

//! @brief Sorts data using Ian Parberry's unstable pairwise sorting network.
//!
//! The algorithm has \f$O(N \log^2 N)\f$ complexity and supports any positive number of items, including non-powers of
//! two.
//!
//! @note The algorithm is particularly useful for order-statistic selection, such as median and top-k.
//! The compiler can eliminate unnecessary operations when only a subset of the output is consumed.
//!
//! @tparam KeyT Key type.
//! @tparam ValueT Value type. If @c cub::NullType is used, only keys are sorted.
//! @tparam CompareOp Comparison function object that provides a strict weak ordering.
//! @tparam ItemPerThread Number of items per thread.
//!
//! @param[in,out] keys Keys to sort.
//! @param[in,out] items Values to reorder with their corresponding keys.
//! @param[in] compare_op Comparison function object.
//!
//! @see Ian Parberry, "The Pairwise Sorting Network", Parallel Processing Letters,
//! Vol. 2, No. 2-3, pp. 205-211, 1992. https://ianparberry.com/pubs/pairwise.pdf
template <typename KeyT, typename ValueT, typename CompareOp, int ItemPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
UnstablePairwiseSort(KeyT (&keys)[ItemPerThread], ValueT (&items)[ItemPerThread], CompareOp compare_op)
{
  return cub::detail::unstable_pairwise_sort(keys, items, compare_op);
}

CUB_NAMESPACE_END
