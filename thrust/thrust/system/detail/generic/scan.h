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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/scan.h>

#include <cuda/functional>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result)
{
  // assume plus as the associative operator
  return thrust::inclusive_scan(exec, first, last, result, ::cuda::std::plus<>());
} // end inclusive_scan()

// Note: it is an error to call this function: this should be provided by a backend system
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op) = delete;

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result)
{
  // Use the input iterator's value type per https://wg21.link/P0571
  using ValueType = thrust::detail::it_value_t<InputIterator>;
  return thrust::exclusive_scan(exec, first, last, result, ValueType{});
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init)
{
  // assume plus as the associative operator
  return thrust::exclusive_scan(exec, first, last, result, init, ::cuda::std::plus<>());
}

// Note: it is an error to call this function: this should be provided by a backend system
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init,
  BinaryFunction binary_op) = delete;
} // namespace system::detail::generic
THRUST_NAMESPACE_END
