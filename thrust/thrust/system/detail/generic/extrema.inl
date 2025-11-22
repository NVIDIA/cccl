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
#include <thrust/detail/get_iterator_value.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace detail
{
//////////////
// Functors //
//////////////
//

// return the smaller/larger element making sure to prefer the
// first occurrence of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct min_element_reduction
{
  BinaryPredicate comp;

  _CCCL_HOST_DEVICE min_element_reduction(BinaryPredicate comp)
      : comp(comp)
  {}

  _CCCL_HOST_DEVICE ::cuda::std::tuple<InputType, IndexType>
  operator()(const ::cuda::std::tuple<InputType, IndexType>& lhs, const ::cuda::std::tuple<InputType, IndexType>& rhs)
  {
    if (comp(::cuda::std::get<0>(lhs), ::cuda::std::get<0>(rhs)))
    {
      return lhs;
    }
    if (comp(::cuda::std::get<0>(rhs), ::cuda::std::get<0>(lhs)))
    {
      return rhs;
    }

    // values are equivalent, prefer value with smaller index
    if (::cuda::std::get<1>(lhs) < ::cuda::std::get<1>(rhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  } // end operator()()
}; // end min_element_reduction

template <typename InputType, typename IndexType, typename BinaryPredicate>
struct max_element_reduction
{
  BinaryPredicate comp;

  _CCCL_HOST_DEVICE max_element_reduction(BinaryPredicate comp)
      : comp(comp)
  {}

  _CCCL_HOST_DEVICE ::cuda::std::tuple<InputType, IndexType>
  operator()(const ::cuda::std::tuple<InputType, IndexType>& lhs, const ::cuda::std::tuple<InputType, IndexType>& rhs)
  {
    if (comp(::cuda::std::get<0>(lhs), ::cuda::std::get<0>(rhs)))
    {
      return rhs;
    }
    if (comp(::cuda::std::get<0>(rhs), ::cuda::std::get<0>(lhs)))
    {
      return lhs;
    }

    // values are equivalent, prefer value with smaller index
    if (::cuda::std::get<1>(lhs) < ::cuda::std::get<1>(rhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  } // end operator()()
}; // end max_element_reduction

// return the smaller & larger element making sure to prefer the
// first occurrence of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct minmax_element_reduction
{
  BinaryPredicate comp;

  _CCCL_HOST_DEVICE minmax_element_reduction(BinaryPredicate comp)
      : comp(comp)
  {}

  _CCCL_HOST_DEVICE ::cuda::std::tuple<::cuda::std::tuple<InputType, IndexType>, ::cuda::std::tuple<InputType, IndexType>>
  operator()(
    const ::cuda::std::tuple<::cuda::std::tuple<InputType, IndexType>, ::cuda::std::tuple<InputType, IndexType>>& lhs,
    const ::cuda::std::tuple<::cuda::std::tuple<InputType, IndexType>, ::cuda::std::tuple<InputType, IndexType>>& rhs)
  {
    return ::cuda::std::make_tuple(
      min_element_reduction<InputType, IndexType, BinaryPredicate>(
        comp)(::cuda::std::get<0>(lhs), ::cuda::std::get<0>(rhs)),
      max_element_reduction<InputType, IndexType, BinaryPredicate>(
        comp)(::cuda::std::get<1>(lhs), ::cuda::std::get<1>(rhs)));
  } // end operator()()
}; // end minmax_element_reduction

template <typename InputType, typename IndexType>
struct duplicate_tuple
{
  _CCCL_HOST_DEVICE ::cuda::std::tuple<::cuda::std::tuple<InputType, IndexType>, ::cuda::std::tuple<InputType, IndexType>>
  operator()(const ::cuda::std::tuple<InputType, IndexType>& t)
  {
    return ::cuda::std::make_tuple(t, t);
  }
}; // end duplicate_tuple
} // end namespace detail

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
min_element(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::min_element(exec, first, last, ::cuda::std::less<value_type>());
} // end min_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator min_element(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  if (first == last)
  {
    return last;
  }

  using InputType = thrust::detail::it_value_t<ForwardIterator>;
  using IndexType = thrust::detail::it_difference_t<ForwardIterator>;

  ::cuda::std::tuple<InputType, IndexType> result = thrust::reduce(
    exec,
    thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)),
    thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)) + (last - first),
    ::cuda::std::tuple<InputType, IndexType>(thrust::detail::get_iterator_value(derived_cast(exec), first), 0),
    detail::min_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + ::cuda::std::get<1>(result);
} // end min_element()

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
max_element(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::max_element(exec, first, last, ::cuda::std::less<value_type>());
} // end max_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator max_element(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  if (first == last)
  {
    return last;
  }

  using InputType = thrust::detail::it_value_t<ForwardIterator>;
  using IndexType = thrust::detail::it_difference_t<ForwardIterator>;

  ::cuda::std::tuple<InputType, IndexType> result = thrust::reduce(
    exec,
    thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)),
    thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)) + (last - first),
    ::cuda::std::tuple<InputType, IndexType>(thrust::detail::get_iterator_value(derived_cast(exec), first), 0),
    detail::max_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return first + ::cuda::std::get<1>(result);
} // end max_element()

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator>
minmax_element(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::minmax_element(exec, first, last, ::cuda::std::less<value_type>());
} // end minmax_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  if (first == last)
  {
    return ::cuda::std::make_pair(last, last);
  }

  using InputType = thrust::detail::it_value_t<ForwardIterator>;
  using IndexType = thrust::detail::it_difference_t<ForwardIterator>;

  ::cuda::std::tuple<::cuda::std::tuple<InputType, IndexType>, ::cuda::std::tuple<InputType, IndexType>> result =
    thrust::transform_reduce(
      exec,
      thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)),
      thrust::make_zip_iterator(first, thrust::counting_iterator<IndexType>(0)) + (last - first),
      detail::duplicate_tuple<InputType, IndexType>(),
      detail::duplicate_tuple<InputType, IndexType>()(
        ::cuda::std::tuple<InputType, IndexType>(thrust::detail::get_iterator_value(derived_cast(exec), first), 0)),
      detail::minmax_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

  return ::cuda::std::make_pair(
    first + ::cuda::std::get<1>(::cuda::std::get<0>(result)), first + ::cuda::std::get<1>(::cuda::std::get<1>(result)));
} // end minmax_element()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
