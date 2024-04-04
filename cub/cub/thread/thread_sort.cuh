/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

enum class stability_t
{
  stable,
  unstable
};

template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void Swap(T& lhs, T& rhs)
{
  const T temp = lhs;
  lhs          = rhs;
  rhs          = temp;
}

namespace detail
{

template <typename IntT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr IntT NetworkDegree(IntT n, IntT m = IntT{1})
{
  return (m < n) ? NetworkDegree(n, m * 2) + 1 : 0;
}

template <typename KeyT>
_CCCL_DEVICE _CCCL_FORCEINLINE void CompareSwapMinMaxAsc(KeyT& key_lhs, KeyT& key_rhs)
{
  const KeyT pair_min = (cub::min)(key_lhs, key_rhs);
  const KeyT pair_max = (cub::max)(key_lhs, key_rhs);
  key_lhs             = pair_min;
  key_rhs             = pair_max;
}

template <typename KeyT, typename ValueT, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void
CompareSwap(KeyT& key_lhs, KeyT& key_rhs, ValueT& item_lhs, ValueT& item_rhs, CompareOp compare_op)
{
  constexpr bool KEYS_ONLY = ::cuda::std::is_same<ValueT, NullType>::value;

  if (compare_op(key_rhs, key_lhs))
  {
    Swap(key_lhs, key_rhs);
    _CCCL_IF_CONSTEXPR (!KEYS_ONLY)
    {
      Swap(item_lhs, item_rhs);
    }
  }
}

#define CUB_SPECIALIZE_SORT_ASC(T)                                                        \
  template <>                                                                             \
  _CCCL_DEVICE _CCCL_FORCEINLINE void CompareSwap(                                        \
    T& key_lhs, T& key_rhs, NullType& item_lhs, NullType& item_rhs, cub::Less compare_op) \
  {                                                                                       \
    CompareSwapMinMaxAsc(key_lhs, key_rhs);                                               \
  }

#define CUB_SPECIALIZE_SORT_DESC(T)                                                          \
  template <>                                                                                \
  _CCCL_DEVICE _CCCL_FORCEINLINE void CompareSwap(                                           \
    T& key_lhs, T& key_rhs, NullType& item_lhs, NullType& item_rhs, cub::Greater compare_op) \
  {                                                                                          \
    CompareSwapMinMaxAsc(key_rhs, key_lhs);                                                  \
  }

CUB_SPECIALIZE_SORT_ASC(::cuda::std::int32_t)
CUB_SPECIALIZE_SORT_ASC(::cuda::std::uint32_t)
CUB_SPECIALIZE_SORT_ASC(float)
CUB_SPECIALIZE_SORT_DESC(::cuda::std::int32_t)
CUB_SPECIALIZE_SORT_DESC(::cuda::std::uint32_t)
CUB_SPECIALIZE_SORT_DESC(float)

#undef CUB_SPECIALIZE_SORT_ASC
#undef CUB_SPECIALIZE_SORT_DESC

} // namespace detail

/**
 * @brief Sorts data using odd-even transposition sort method
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
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
 *   `CompareOp` is a model of [Strict Weak Ordering].
 *
 * @tparam ITEMS_PER_THREAD
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
 *
 * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StableOddEvenSort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
#pragma unroll
    for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
    {
      detail::CompareSwap(keys[j], keys[j + 1], items[j], items[j + 1], compare_op);
    } // inner loop
  } // outer loop
}

/**
 * @brief Sorts data using Ian Parberry's pairwise sorting network
 *
 * The sorting method is not stable. Further details can be found in:
 * Ian Parberry, "The Pairwise Sorting Network". Parallel Processing Letters,
 * Vol. 2, No. 2,3, pp. 205-211, 1992.
 * https://ianparberry.com/pubs/pairwise.pdf
 *
 * The complexity is O(N log^2(N)), as opposed to O(N^2) for the stable odd-even
 * transposition sort. E.g. for N=16, pairwise is 2x faster; for N=64, 4x faster.
 *
 * This method generally outperforms Batcher's odd-even mergesort, especially
 * for selection (top-k or median), as larger parts of the sorting network can be
 * pruned by compiler optimizations.
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type. If `cub::NullType` is used as `ValueT`, only keys are sorted.
 *
 * @tparam CompareOp
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
 *   `CompareOp` is a model of [Strict Weak Ordering].
 *
 * @tparam ITEMS_PER_THREAD
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
 *
 * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
PairwiseSort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
  constexpr int N = ITEMS_PER_THREAD;
  constexpr int D = detail::NetworkDegree(N);
  constexpr int M = 1 << D; // Smallest power of two greater than N

#pragma unroll
  for (int i = 1; i < M; i *= 2)
  {
#pragma unroll
    for (int j = 0; j < i; j++)
    {
#pragma unroll
      for (int idx_lhs = j; idx_lhs < N - i; idx_lhs += 2 * i)
      {
        const int idx_rhs = idx_lhs + i;
        detail::CompareSwap(keys[idx_lhs], keys[idx_rhs], items[idx_lhs], items[idx_rhs], compare_op);
      }
    }
  }

#pragma unroll
  for (int p = 0; p < D - 1; p++)
  {
    const int i = M >> (p + 2);
    const int k = (1 << (p + 1)) - 1;
#pragma unroll
    for (int j = k; j > 0; j /= 2)
    {
      const int delta = i * j;
#pragma unroll
      for (int idx_lhs = 0; idx_lhs < N; idx_lhs++)
      {
        if ((idx_lhs / i) % 2 == 1)
        {
          const int idx_rhs = idx_lhs + delta;
          if (idx_rhs < N)
          {
            detail::CompareSwap(keys[idx_lhs], keys[idx_rhs], items[idx_lhs], items[idx_rhs], compare_op);
          }
        }
      }
    }
  }
}

/**
 * @brief Sorts data using a sorting network
 *
 * Wraps around stable and unstable methods. When stability is not required, the unstable sort offers better
 * performance.
 *
 * For all methods, it is valid to pad `keys` to the right with a key M such that there is no
 * key K in `keys` that satisfies compare_op(M, K).
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type. If `cub::NullType` is used as `ValueT`, only keys are sorted.
 *
 * @tparam CompareOp
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`.
 *   `CompareOp` is a model of [Strict Weak Ordering].
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of items per thread
 *
 * @tparam Stability
 *   Whether to use a stable sorting method.
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
 *
 * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD, stability_t Stability>
_CCCL_DEVICE _CCCL_FORCEINLINE void
ThreadNetworkSort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
  _CCCL_IF_CONSTEXPR (Stability == stability_t::stable)
  {
    StableOddEvenSort(keys, items, compare_op);
  }
  else
  {
    PairwiseSort(keys, items, compare_op);
  }
}

CUB_NAMESPACE_END
