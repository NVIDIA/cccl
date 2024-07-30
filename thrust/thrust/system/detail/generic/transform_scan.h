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

_CCCL_IMPLICIT_SYSTEM_HEADER
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform_inclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  BinaryFunction binary_op);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator transform_exclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op);

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/transform_scan.inl>
