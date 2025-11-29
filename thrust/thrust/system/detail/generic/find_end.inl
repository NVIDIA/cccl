/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/find_end.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator1 find_end(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2,
  ForwardIterator2 last2)
{
  return thrust::find_end(exec, first1, last1, first2, last2, ::cuda::std::equal_to<>());
}

template <typename TupleType>
struct find_end_reduce_functor
{
  _CCCL_HOST_DEVICE TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (thrust::get<0>(lhs) && thrust::get<0>(rhs))
    {
      return TupleType(true, (::cuda::std::max) (thrust::get<1>(lhs), thrust::get<1>(rhs)));
    }
    else if (thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

// Functor that checks whether pattern [first2, last2) matches
// starting at position i in [first1, last1).
template <typename ForwardIterator1, typename ForwardIterator2, typename difference_type, typename BinaryPredicate>
struct find_end_match_functor
{
  ForwardIterator1 first1;
  ForwardIterator2 first2;
  difference_type n2;
  BinaryPredicate pred;
  using result_type = thrust::tuple<bool, difference_type>;

  _CCCL_HOST_DEVICE
  find_end_match_functor(ForwardIterator1 a, ForwardIterator2 b, difference_type len, BinaryPredicate p)
      : first1(a)
      , first2(b)
      , n2(len)
      , pred(p)
  {}

  _CCCL_HOST_DEVICE result_type operator()(difference_type i) const
  {
    for (difference_type j = 0; j < n2; ++j)
    {
      if (!pred(*(first1 + (i + j)), *(first2 + j)))
      {
        return thrust::make_tuple(false, i);
      }
    }
    return thrust::make_tuple(true, i);
  }
};

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 find_end(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2,
  ForwardIterator2 last2,
  BinaryPredicate pred)
{
  using difference_type = thrust::detail::it_difference_t<ForwardIterator1>;
  using result_type     = thrust::tuple<bool, difference_type>;

  if (first2 == last2)
  {
    return last1;
  }

  const difference_type n1 = ::cuda::std::distance(first1, last1);
  const difference_type n2 = ::cuda::std::distance(first2, last2);

  if (n1 < n2)
  {
    return last1;
  }

  using MatchType = find_end_match_functor<ForwardIterator1, ForwardIterator2, difference_type, BinaryPredicate>;
  MatchType match_at_index(first1, first2, n2, pred);

  // Build transform iterator over counting_iterator(0..n1-n2)
  auto count_begin     = thrust::counting_iterator<difference_type>(0);
  auto transform_begin = thrust::make_transform_iterator(count_begin, match_at_index);
  auto transform_end   = transform_begin + (n1 - n2 + 1);

  // If no match found, index set to -1 (signed difference_type)
  result_type init = thrust::make_tuple(false, static_cast<difference_type>(-1));

  // reduce across transformed values to pick the last match
  result_type result =
    thrust::reduce(exec, transform_begin, transform_end, init, find_end_reduce_functor<result_type>());

  if (thrust::get<0>(result))
  {
    // found: return iterator to start position
    return first1 + thrust::get<1>(result);
  }

  return last1;
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
