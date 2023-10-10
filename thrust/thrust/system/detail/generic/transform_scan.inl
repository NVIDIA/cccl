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
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/transform_scan.h>
#include <thrust/type_traits/remove_cvref.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename BinaryFunction>
__host__ __device__
  OutputIterator transform_inclusive_scan(thrust::execution_policy<ExecutionPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          BinaryFunction binary_op)
{
  // Use the input iterator's value type per https://wg21.link/P0571
  using InputType = typename thrust::iterator_value<InputIterator>::type;
  using ResultType = thrust::detail::invoke_result_t<UnaryFunction, InputType>;
  using ValueType = thrust::remove_cvref_t<ResultType>;

  thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return thrust::inclusive_scan(exec, _first, _last, result, binary_op);
} // end transform_inclusive_scan()


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename InitialValueType,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator transform_exclusive_scan(thrust::execution_policy<ExecutionPolicy> &exec,
                                          InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          UnaryFunction unary_op,
                                          InitialValueType init,
                                          AssociativeOperator binary_op)
{
  // Use the initial value type per https://wg21.link/P0571
  using ValueType = thrust::remove_cvref_t<InitialValueType>;

  thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

  return thrust::exclusive_scan(exec, _first, _last, result, init, binary_op);
} // end transform_exclusive_scan()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

