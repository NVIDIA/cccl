// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/execution_policy.h>
#include <thrust/distance.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/iterator>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
// Functor that checks if the pattern matches starting at a given position
template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
struct pattern_match
{
  ForwardIterator1 haystack_first;
  ForwardIterator1 haystack_last;
  ForwardIterator2 needle_first;
  ForwardIterator2 needle_last;
  BinaryPredicate pred;

  _CCCL_HOST_DEVICE pattern_match(
    ForwardIterator1 haystack_first,
    ForwardIterator1 haystack_last,
    ForwardIterator2 needle_first,
    ForwardIterator2 needle_last,
    BinaryPredicate pred)
      : haystack_first(haystack_first)
      , haystack_last(haystack_last)
      , needle_first(needle_first)
      , needle_last(needle_last)
      , pred(pred)
  {}

  template <typename Index>
  _CCCL_HOST_DEVICE bool operator()(Index idx) const
  {
    ForwardIterator1 haystack_pos = haystack_first;
    ::cuda::std::advance(haystack_pos, idx);

    // Check if we have enough elements remaining
    auto remaining = ::cuda::std::distance(haystack_pos, haystack_last);
    auto needed    = ::cuda::std::distance(needle_first, needle_last);
    if (remaining < needed)
    {
      return false;
    }

    // Check if all elements match
    ForwardIterator1 h_it = haystack_pos;
    ForwardIterator2 n_it = needle_first;
    for (; n_it != needle_last; ++h_it, ++n_it)
    {
      if (!pred(*h_it, *n_it))
      {
        return false;
      }
    }
    return true;
  }
};

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last,
  BinaryPredicate pred)
{
  using difference_type = thrust::detail::it_difference_t<ForwardIterator1>;

  // Empty pattern is found at the beginning
  if (s_first == s_last)
  {
    return first;
  }

  // Empty haystack cannot contain the pattern
  if (first == last)
  {
    return last;
  }

  const difference_type haystack_size = ::cuda::std::distance(first, last);
  const difference_type needle_size   = ::cuda::std::distance(s_first, s_last);

  if (needle_size > haystack_size)
  {
    return last;
  }

  // Maximum number of positions to check
  const difference_type search_size = haystack_size - needle_size + 1;

  // Create a predicate that checks if pattern matches at each position
  pattern_match<ForwardIterator1, ForwardIterator2, BinaryPredicate> match_pred(first, last, s_first, s_last, pred);

  // Use find_if to search in parallel for the first matching position
  auto counting_first = thrust::counting_iterator<difference_type>(0);
  auto counting_last  = counting_first + search_size;

  auto result_iter = thrust::find_if(exec, counting_first, counting_last, match_pred);

  // If found, advance to that position; otherwise return last
  if (result_iter != counting_last)
  {
    ForwardIterator1 result = first;
    ::cuda::std::advance(result, *result_iter);
    return result;
  }

  return last;
}

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator1>;

  return thrust::system::detail::generic::search(
    exec, first, last, s_first, s_last, ::cuda::std::equal_to<value_type>());
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
