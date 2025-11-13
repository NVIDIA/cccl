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
#include <thrust/detail/static_assert.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/set_operations.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_difference(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
} // end set_difference()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_difference_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    ::cuda::std::less<value_type>());
} // end set_difference_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  using iterator_tuple1 = thrust::tuple<InputIterator1, InputIterator3>;
  using iterator_tuple2 = thrust::tuple<InputIterator2, InputIterator4>;
  using iterator_tuple3 = thrust::tuple<OutputIterator1, OutputIterator2>;

  using zip_iterator1 = thrust::zip_iterator<iterator_tuple1>;
  using zip_iterator2 = thrust::zip_iterator<iterator_tuple2>;
  using zip_iterator3 = thrust::zip_iterator<iterator_tuple3>;

  zip_iterator1 zipped_first1 = thrust::make_zip_iterator(keys_first1, values_first1);
  zip_iterator1 zipped_last1  = thrust::make_zip_iterator(keys_last1, values_first1);

  zip_iterator2 zipped_first2 = thrust::make_zip_iterator(keys_first2, values_first2);
  zip_iterator2 zipped_last2  = thrust::make_zip_iterator(keys_last2, values_first2);

  zip_iterator3 zipped_result = thrust::make_zip_iterator(keys_result, values_result);

  thrust::detail::compare_first<StrictWeakOrdering> comp_first{comp};

  iterator_tuple3 result =
    thrust::set_difference(exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first)
      .get_iterator_tuple();

  return ::cuda::std::make_pair(thrust::get<0>(result), thrust::get<1>(result));
} // end set_difference_by_key()

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_intersection(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
} // end set_intersection()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_intersection_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    keys_result,
    values_result,
    ::cuda::std::less<value_type>());
} // end set_intersection_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  using value_type1       = thrust::detail::it_value_t<InputIterator3>;
  using constant_iterator = thrust::constant_iterator<value_type1>;

  using iterator_tuple1 = thrust::tuple<InputIterator1, InputIterator3>;
  using iterator_tuple2 = thrust::tuple<InputIterator2, constant_iterator>;
  using iterator_tuple3 = thrust::tuple<OutputIterator1, OutputIterator2>;

  using zip_iterator1 = thrust::zip_iterator<iterator_tuple1>;
  using zip_iterator2 = thrust::zip_iterator<iterator_tuple2>;
  using zip_iterator3 = thrust::zip_iterator<iterator_tuple3>;

  // fabricate a values_first2 by repeating a default-constructed value_type1
  // XXX assumes value_type1 is default-constructible
  constant_iterator values_first2 = thrust::make_constant_iterator(value_type1());

  zip_iterator1 zipped_first1 = thrust::make_zip_iterator(keys_first1, values_first1);
  zip_iterator1 zipped_last1  = thrust::make_zip_iterator(keys_last1, values_first1);

  zip_iterator2 zipped_first2 = thrust::make_zip_iterator(keys_first2, values_first2);
  zip_iterator2 zipped_last2  = thrust::make_zip_iterator(keys_last2, values_first2);

  zip_iterator3 zipped_result = thrust::make_zip_iterator(keys_result, values_result);

  thrust::detail::compare_first<StrictWeakOrdering> comp_first{comp};

  iterator_tuple3 result =
    thrust::set_intersection(exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first)
      .get_iterator_tuple();

  return ::cuda::std::make_pair(thrust::get<0>(result), thrust::get<1>(result));
} // end set_intersection_by_key()

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_symmetric_difference(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
} // end set_symmetric_difference()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_symmetric_difference_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    ::cuda::std::less<value_type>());
} // end set_symmetric_difference_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  using iterator_tuple1 = thrust::tuple<InputIterator1, InputIterator3>;
  using iterator_tuple2 = thrust::tuple<InputIterator2, InputIterator4>;
  using iterator_tuple3 = thrust::tuple<OutputIterator1, OutputIterator2>;

  using zip_iterator1 = thrust::zip_iterator<iterator_tuple1>;
  using zip_iterator2 = thrust::zip_iterator<iterator_tuple2>;
  using zip_iterator3 = thrust::zip_iterator<iterator_tuple3>;

  zip_iterator1 zipped_first1 = thrust::make_zip_iterator(keys_first1, values_first1);
  zip_iterator1 zipped_last1  = thrust::make_zip_iterator(keys_last1, values_first1);

  zip_iterator2 zipped_first2 = thrust::make_zip_iterator(keys_first2, values_first2);
  zip_iterator2 zipped_last2  = thrust::make_zip_iterator(keys_last2, values_first2);

  zip_iterator3 zipped_result = thrust::make_zip_iterator(keys_result, values_result);

  thrust::detail::compare_first<StrictWeakOrdering> comp_first{comp};

  iterator_tuple3 result =
    thrust::set_symmetric_difference(
      exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first)
      .get_iterator_tuple();

  return ::cuda::std::make_pair(thrust::get<0>(result), thrust::get<1>(result));
} // end set_symmetric_difference_by_key()

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_union(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_union(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
} // end set_union()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return thrust::set_union_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    ::cuda::std::less<value_type>());
} // end set_union_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  using iterator_tuple1 = thrust::tuple<InputIterator1, InputIterator3>;
  using iterator_tuple2 = thrust::tuple<InputIterator2, InputIterator4>;
  using iterator_tuple3 = thrust::tuple<OutputIterator1, OutputIterator2>;

  using zip_iterator1 = thrust::zip_iterator<iterator_tuple1>;
  using zip_iterator2 = thrust::zip_iterator<iterator_tuple2>;
  using zip_iterator3 = thrust::zip_iterator<iterator_tuple3>;

  zip_iterator1 zipped_first1 = thrust::make_zip_iterator(keys_first1, values_first1);
  zip_iterator1 zipped_last1  = thrust::make_zip_iterator(keys_last1, values_first1);

  zip_iterator2 zipped_first2 = thrust::make_zip_iterator(keys_first2, values_first2);
  zip_iterator2 zipped_last2  = thrust::make_zip_iterator(keys_last2, values_first2);

  zip_iterator3 zipped_result = thrust::make_zip_iterator(keys_result, values_result);

  thrust::detail::compare_first<StrictWeakOrdering> comp_first{comp};

  iterator_tuple3 result =
    thrust::set_union(exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first)
      .get_iterator_tuple();

  return ::cuda::std::make_pair(thrust::get<0>(result), thrust::get<1>(result));
} // end set_union_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  thrust::execution_policy<DerivedPolicy>&,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  OutputIterator result,
  StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator1, false>::value, "unimplemented for this system");
  return result;
} // end set_difference()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  thrust::execution_policy<DerivedPolicy>&,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  OutputIterator result,
  StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator1, false>::value, "unimplemented for this system");
  return result;
} // end set_intersection()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  thrust::execution_policy<DerivedPolicy>&,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  OutputIterator result,
  StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator1, false>::value, "unimplemented for this system");
  return result;
} // end set_symmetric_difference()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_union(
  thrust::execution_policy<DerivedPolicy>&,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  OutputIterator result,
  StrictWeakOrdering)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator1, false>::value, "unimplemented for this system");
  return result;
} // end set_union()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
