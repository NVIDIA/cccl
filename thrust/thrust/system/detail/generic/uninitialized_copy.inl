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
#include <thrust/copy.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/uninitialized_copy.h>

#include <cuda/std/__new/device_new.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace detail
{
template <typename InputType, typename OutputType>
struct uninitialized_copy_functor
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    const InputType& in = thrust::get<0>(t);
    OutputType& out     = thrust::get<1>(t);
    ::new (static_cast<void*>(&out)) OutputType(in);
  } // end operator()()
}; // end uninitialized_copy_functor

// non-trivial copy constructor path
template <typename ExecutionPolicy, typename InputIterator, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  ForwardIterator result,
  thrust::detail::false_type) // ::cuda::std::is_trivially_copy_constructible
{
  // zip up the iterators
  using IteratorTuple = thrust::tuple<InputIterator, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator begin = thrust::make_zip_iterator(first, result);
  ZipIterator end   = begin;

  // get a zip_iterator pointing to the end
  const thrust::detail::it_difference_t<InputIterator> n = ::cuda::std::distance(first, last);
  ::cuda::std::advance(end, n);

  // create a functor
  using InputType  = thrust::detail::it_value_t<InputIterator>;
  using OutputType = thrust::detail::it_value_t<ForwardIterator>;

  detail::uninitialized_copy_functor<InputType, OutputType> f;

  // do the for_each
  thrust::for_each(exec, begin, end, f);

  // return the end of the output range
  return thrust::get<1>(end.get_iterator_tuple());
} // end uninitialized_copy()

// trivial copy constructor path
template <typename ExecutionPolicy, typename InputIterator, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  ForwardIterator result,
  thrust::detail::true_type) // ::cuda::std::is_trivially_copy_constructible
{
  return thrust::copy(exec, first, last, result);
} // end uninitialized_copy()

// non-trivial copy constructor path
template <typename ExecutionPolicy, typename InputIterator, typename Size, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy_n(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  Size n,
  ForwardIterator result,
  thrust::detail::false_type) // ::cuda::std::is_trivially_copy_constructible
{
  // zip up the iterators
  using IteratorTuple = thrust::tuple<InputIterator, ForwardIterator>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator zipped_first = thrust::make_zip_iterator(first, result);

  // create a functor
  using InputType  = thrust::detail::it_value_t<InputIterator>;
  using OutputType = thrust::detail::it_value_t<ForwardIterator>;

  detail::uninitialized_copy_functor<InputType, OutputType> f;

  // do the for_each_n
  ZipIterator zipped_last = thrust::for_each_n(exec, zipped_first, n, f);

  // return the end of the output range
  return thrust::get<1>(zipped_last.get_iterator_tuple());
} // end uninitialized_copy_n()

// trivial copy constructor path
template <typename ExecutionPolicy, typename InputIterator, typename Size, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy_n(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  Size n,
  ForwardIterator result,
  thrust::detail::true_type) // ::cuda::std::is_trivially_copy_constructible
{
  return thrust::copy_n(exec, first, n, result);
} // end uninitialized_copy_n()
} // namespace detail

template <typename ExecutionPolicy, typename InputIterator, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy(
  thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, ForwardIterator result)
{
  using ResultType = thrust::detail::it_value_t<ForwardIterator>;

  using ResultTypeHasTrivialCopyConstructor = typename ::cuda::std::is_trivially_copy_constructible<ResultType>::type;

  return thrust::system::detail::generic::detail::uninitialized_copy(
    exec, first, last, result, ResultTypeHasTrivialCopyConstructor());
} // end uninitialized_copy()

template <typename ExecutionPolicy, typename InputIterator, typename Size, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_copy_n(
  thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, Size n, ForwardIterator result)
{
  using ResultType = thrust::detail::it_value_t<ForwardIterator>;

  using ResultTypeHasTrivialCopyConstructor = typename ::cuda::std::is_trivially_copy_constructible<ResultType>::type;

  return thrust::system::detail::generic::detail::uninitialized_copy_n(
    exec, first, n, result, ResultTypeHasTrivialCopyConstructor());
} // end uninitialized_copy_n()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
