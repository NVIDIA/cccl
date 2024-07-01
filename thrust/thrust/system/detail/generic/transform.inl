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
#include <thrust/for_each.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/transform.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction op)
{
  using UnaryTransformFunctor = thrust::detail::unary_transform_functor<UnaryFunction>;

  // make an iterator tuple
  using IteratorTuple = thrust::tuple<InputIterator, OutputIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(first, result)),
    thrust::make_zip_iterator(thrust::make_tuple(last, result)),
    UnaryTransformFunctor(op));

  return thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  // given the minimal system, determine the binary transform functor we need
  using BinaryTransformFunctor = thrust::detail::binary_transform_functor<BinaryFunction>;

  // make an iterator tuple
  using IteratorTuple = thrust::tuple<InputIterator1, InputIterator2, OutputIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(first1, first2, result)),
    thrust::make_zip_iterator(thrust::make_tuple(last1, first2, result)),
    BinaryTransformFunctor(op));

  return thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform()

template <typename DerivedPolicy,
          typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  ForwardIterator result,
  UnaryFunction unary_op,
  Predicate pred)
{
  using UnaryTransformIfFunctor = thrust::detail::unary_transform_if_functor<UnaryFunction, Predicate>;

  // make an iterator tuple
  using IteratorTuple = thrust::tuple<InputIterator, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(first, result)),
    thrust::make_zip_iterator(thrust::make_tuple(last, result)),
    UnaryTransformIfFunctor(unary_op, pred));

  return thrust::get<1>(zipped_result.get_iterator_tuple());
} // end transform_if()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction unary_op,
  Predicate pred)
{
  using UnaryTransformIfFunctor = thrust::detail::unary_transform_if_with_stencil_functor<UnaryFunction, Predicate>;

  // make an iterator tuple
  using IteratorTuple = thrust::tuple<InputIterator1, InputIterator2, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(first, stencil, result)),
    thrust::make_zip_iterator(thrust::make_tuple(last, stencil, result)),
    UnaryTransformIfFunctor(unary_op, pred));

  return thrust::get<2>(zipped_result.get_iterator_tuple());
} // end transform_if()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred)
{
  using BinaryTransformIfFunctor = thrust::detail::binary_transform_if_functor<BinaryFunction, Predicate>;

  // make an iterator tuple
  using IteratorTuple = thrust::tuple<InputIterator1, InputIterator2, InputIterator3, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(first1, first2, stencil, result)),
    thrust::make_zip_iterator(thrust::make_tuple(last1, first2, stencil, result)),
    BinaryTransformIfFunctor(binary_op, pred));

  return thrust::get<3>(zipped_result.get_iterator_tuple());
} // end transform_if()

} // namespace generic
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END
