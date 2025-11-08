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
#include <thrust/detail/copy_if.h>
#include <thrust/detail/count.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/unique.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <cuda/std/__functional/operations.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
unique(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::unique(exec, first, last, ::cuda::std::equal_to<InputType>());
} // end unique()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator unique(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  thrust::detail::temporary_array<InputType, DerivedPolicy> input(exec, first, last);

  return thrust::unique_copy(exec, input.begin(), input.end(), first, binary_pred);
} // end unique()

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator output)
{
  using value_type = thrust::detail::it_value_t<InputIterator>;
  return thrust::unique_copy(exec, first, last, output, ::cuda::std::equal_to<value_type>());
} // end unique_copy()

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred)
{
  thrust::detail::head_flags<InputIterator, BinaryPredicate> stencil(first, last, binary_pred);

  using namespace thrust::placeholders;

  return thrust::copy_if(exec, first, last, stencil.begin(), output, _1);
} // end unique_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator> unique_count(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred)
{
  thrust::detail::head_flags<ForwardIterator, BinaryPredicate> stencil(first, last, binary_pred);

  using namespace thrust::placeholders;

  return thrust::count_if(exec, stencil.begin(), stencil.end(), _1);
} // end unique_copy()

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator>
unique_count(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return thrust::unique_count(exec, first, last, ::cuda::std::equal_to<value_type>());
} // end unique_copy()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
