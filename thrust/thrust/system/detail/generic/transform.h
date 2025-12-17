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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/tag.h>
#include <thrust/system/detail/generic/transform.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN

// forward declare thrust API entry points to which the generic implementation delegates. We cannot #include
// <thrust/transform.h>, since that header already includes this file.

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction op);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op);

template <typename DerivedPolicy,
          typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred);

namespace system::detail::generic
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
    exec, thrust::make_zip_iterator(first, result), thrust::make_zip_iterator(last, result), UnaryTransformFunctor{op});

  return thrust::get<1>(zipped_result.get_iterator_tuple());
}

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
    thrust::make_zip_iterator(first1, first2, result),
    thrust::make_zip_iterator(last1, first2, result),
    BinaryTransformFunctor{op});

  return thrust::get<2>(zipped_result.get_iterator_tuple());
}

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
    thrust::make_zip_iterator(first, result),
    thrust::make_zip_iterator(last, result),
    UnaryTransformIfFunctor{unary_op, pred});

  return thrust::get<1>(zipped_result.get_iterator_tuple());
}

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
    thrust::make_zip_iterator(first, stencil, result),
    thrust::make_zip_iterator(last, stencil, result),
    UnaryTransformIfFunctor{unary_op, pred});

  return thrust::get<2>(zipped_result.get_iterator_tuple());
}

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
    thrust::make_zip_iterator(first1, first2, stencil, result),
    thrust::make_zip_iterator(last1, first2, stencil, result),
    BinaryTransformIfFunctor{binary_op, pred});

  return thrust::get<3>(zipped_result.get_iterator_tuple());
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  ::cuda::std::iter_difference_t<InputIterator> count,
  OutputIterator result,
  UnaryFunction op)
{
  return thrust::transform(exec, first, first + count, result, op);
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform_n(
  InputIterator first, ::cuda::std::iter_difference_t<InputIterator> count, OutputIterator result, UnaryFunction op)
{
  return thrust::transform(first, first + count, result, op);
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  return thrust::transform(exec, first1, first1 + count, first2, result, op);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
OutputIterator transform_n(
  InputIterator1 first1,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  return thrust::transform(first1, first1 + count, first2, result, op);
}

template <typename DerivedPolicy,
          typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  ::cuda::std::iter_difference_t<InputIterator> count,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  return thrust::transform_if(exec, first, first + count, result, op, pred);
}

template <typename InputIterator, typename ForwardIterator, typename UnaryFunction, typename Predicate>
ForwardIterator transform_if_n(
  InputIterator first,
  ::cuda::std::iter_difference_t<InputIterator> count,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  return thrust::transform_if(first, first + count, result, op, pred);
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  return thrust::transform_if(exec, first, first + count, stencil, result, op, pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
ForwardIterator transform_if_n(
  InputIterator1 first,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  return thrust::transform_if(first, first + count, stencil, result, op, pred);
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred)
{
  return thrust::transform_if(exec, first1, first1 + count, first2, stencil, result, binary_op, pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
ForwardIterator transform_if_n(
  InputIterator1 first1,
  ::cuda::std::iter_difference_t<InputIterator1> count,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred)
{
  return thrust::transform_if(first1, first1 + count, first2, stencil, result, binary_op, pred);
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
