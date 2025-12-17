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
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/sort.h>
#include <thrust/tuple.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
sort(thrust::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator>;
  thrust::sort(exec, first, last, ::cuda::std::less<value_type>());
} // end sort()

template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void
sort(thrust::execution_policy<DerivedPolicy>& exec,
     RandomAccessIterator first,
     RandomAccessIterator last,
     StrictWeakOrdering comp)
{
  // implement with stable_sort
  thrust::stable_sort(exec, first, last, comp);
} // end sort()

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
_CCCL_HOST_DEVICE void sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator1>;
  thrust::sort_by_key(exec, keys_first, keys_last, values_first, ::cuda::std::less<value_type>());
} // end sort_by_key()

template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first,
  StrictWeakOrdering comp)
{
  // implement with stable_sort_by_key
  thrust::stable_sort_by_key(exec, keys_first, keys_last, values_first, comp);
} // end sort_by_key()

template <typename DerivedPolicy, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
stable_sort(thrust::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator>;
  thrust::stable_sort(exec, first, last, ::cuda::std::less<value_type>());
} // end stable_sort()

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
_CCCL_HOST_DEVICE void stable_sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator1>;
  thrust::stable_sort_by_key(exec, keys_first, keys_last, values_first, ::cuda::std::less<value_type>());
} // end stable_sort_by_key()

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE bool
is_sorted(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  return thrust::is_sorted_until(exec, first, last) == last;
} // end is_sorted()

template <typename DerivedPolicy, typename ForwardIterator, typename Compare>
_CCCL_HOST_DEVICE bool
is_sorted(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Compare comp)
{
  return thrust::is_sorted_until(exec, first, last, comp) == last;
} // end is_sorted()

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
is_sorted_until(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::is_sorted_until(exec, first, last, ::cuda::std::less<InputType>());
} // end is_sorted_until()

template <typename DerivedPolicy, typename ForwardIterator, typename Compare>
_CCCL_HOST_DEVICE ForwardIterator is_sorted_until(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Compare comp)
{
  if (::cuda::std::distance(first, last) < 2)
  {
    return last;
  }

  using IteratorTuple = thrust::tuple<ForwardIterator, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ForwardIterator first_plus_one = first;
  ::cuda::std::advance(first_plus_one, 1);

  ZipIterator zipped_first = thrust::make_zip_iterator(first_plus_one, first);
  ZipIterator zipped_last  = thrust::make_zip_iterator(last, first);

  return thrust::get<0>(
    thrust::find_if(exec, zipped_first, zipped_last, thrust::detail::tuple_binary_predicate<Compare>{comp})
      .get_iterator_tuple());
} // end is_sorted_until()

template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void
stable_sort(thrust::execution_policy<DerivedPolicy>&, RandomAccessIterator, RandomAccessIterator, StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value,
                "unimplemented for this system");
} // end stable_sort()

template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void stable_sort_by_key(
  thrust::execution_policy<DerivedPolicy>&,
  RandomAccessIterator1,
  RandomAccessIterator1,
  RandomAccessIterator2,
  StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<RandomAccessIterator1, false>::value,
                "unimplemented for this system");
} // end stable_sort_by_key()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
