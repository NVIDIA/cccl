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

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/readable_traits.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename InputIterator,
          typename InitType       = ::cuda::std::iter_value_t<InputIterator>,
          typename BinaryFunction = ::cuda::std::plus<>>
using __iter_accumulator_t =
  ::cuda::std::__accumulator_t<BinaryFunction, ::cuda::std::iter_value_t<InputIterator>, InitType>;

template <typename InputIterator, typename InitType, typename UnaryFunction, typename BinaryFunction>
using __iter_unary_accumulator_t =
  ::cuda::std::__accumulator_t<BinaryFunction,
                               ::cuda::std::invoke_result_t<UnaryFunction, ::cuda::std::iter_value_t<InputIterator>>,
                               InitType>;

template <typename InputIterator1,
          typename InputIterator2,
          typename InitType        = ::cuda::std::iter_value_t<InputIterator1>,
          typename BinaryFunction1 = ::cuda::std::plus<>,
          typename BinaryFunction2 = ::cuda::std::multiplies<>>
using __inner_product_accumulator_t =
  ::cuda::std::__accumulator_t<BinaryFunction1,
                               ::cuda::std::__accumulator_t<BinaryFunction2,
                                                            ::cuda::std::iter_value_t<InputIterator1>,
                                                            ::cuda::std::iter_value_t<InputIterator2>>,
                               InitType>;

} // namespace detail
THRUST_NAMESPACE_END
