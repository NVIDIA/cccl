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
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/iterator>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last,
  BinaryPredicate pred)
{
  (void) exec;

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

  ForwardIterator1 search_end = first;
  ::cuda::std::advance(search_end, haystack_size - needle_size + 1);

  // Search for the first occurrence
  for (ForwardIterator1 it = first; it != search_end; ++it)
  {
    ForwardIterator1 it1 = it;
    ForwardIterator2 it2 = s_first;

    for (; it2 != s_last; ++it1, ++it2)
    {
      if (!pred(*it1, *it2))
      {
        break;
      }
    }

    if (it2 == s_last)
    {
      return it;
    }
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
