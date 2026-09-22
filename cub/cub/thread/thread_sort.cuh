// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/swap.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <bool Unroll = true, typename KeyT, typename ValueT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
stable_odd_even_sort(KeyT (&keys)[ItemsPerThread], ValueT (&items)[ItemsPerThread], CompareOp compare_op)
{
  constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  _CCCL_PRAGMA_UNROLL(Unroll ? ItemsPerThread : 1)
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    _CCCL_PRAGMA_UNROLL(Unroll ? ItemsPerThread : 1) // unroll count is higher than loop count, but that's fine
    for (int j = 1 & i; j < ItemsPerThread - 1; j += 2)
    {
      if (compare_op(keys[j + 1], keys[j]))
      {
        using ::cuda::std::swap;
        swap(keys[j], keys[j + 1]);
        if constexpr (!KEYS_ONLY)
        {
          swap(items[j], items[j + 1]);
        }
      }
    } // inner loop
  } // outer loop
}
} // namespace detail

/**
 * @brief Sorts data using odd-even sort method
 *
 * The sorting method is stable. Further details can be found in:
 * A. Nico Habermann. Parallel neighbor sort (or the glory of the induction
 * principle). Technical Report AD-759 248, Carnegie Mellon University, 1972.
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
 * @tparam ItemsPerThread
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
template <typename KeyT, typename ValueT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StableOddEvenSort(KeyT (&keys)[ItemsPerThread], ValueT (&items)[ItemsPerThread], CompareOp compare_op)
{
  return detail::stable_odd_even_sort(keys, items, compare_op);
}

CUB_NAMESPACE_END
