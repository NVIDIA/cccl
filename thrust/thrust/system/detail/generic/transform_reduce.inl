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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/transform_reduce.h>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/decay.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator, typename UnaryFunction, typename InitType, typename BinaryFunction>
_CCCL_HOST_DEVICE ::cuda::std::__accumulator_t<
  BinaryFunction,
  decltype(UnaryFunction{}(::cuda::std::declval<::cuda::std::iter_value_t<InputIterator>>())),
  decltype(UnaryFunction{}(::cuda::std::declval<InitType>()))>
transform_reduce(thrust::execution_policy<DerivedPolicy>& exec,
                 InputIterator first,
                 InputIterator last,
                 UnaryFunction unary_op,
                 InitType init,
                 BinaryFunction binary_op)
{
  using AccType = ::cuda::std::__accumulator_t<
    BinaryFunction,
    decltype(UnaryFunction(::cuda::std::declval<::cuda::std::iter_value_t<InputIterator>>())),
    decltype(UnaryFunction(::cuda::std::declval<InitType>()))>;
  thrust::transform_iterator<UnaryFunction, InputIterator, AccType> xfrm_first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, AccType> xfrm_last(last, unary_op);

  return thrust::reduce(exec, xfrm_first, xfrm_last, init, binary_op);
} // end transform_reduce()

} // namespace generic
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END
