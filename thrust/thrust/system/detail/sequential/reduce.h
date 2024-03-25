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


/*! \file reduce.h
 *  \brief Sequential implementation of reduce algorithm.
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
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


_CCCL_EXEC_CHECK_DISABLE
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputType,
         typename BinaryFunction>
_CCCL_HOST_DEVICE
  OutputType reduce(sequential::execution_policy<DerivedPolicy> &,
                    InputIterator begin,
                    InputIterator end,
                    OutputType init,
                    BinaryFunction binary_op)
{
  // wrap binary_op
  thrust::detail::wrapped_function<
    BinaryFunction,
    OutputType
  > wrapped_binary_op(binary_op);

  // initialize the result
  OutputType result = init;

  while(begin != end)
  {
    result = wrapped_binary_op(result, *begin);
    ++begin;
  } // end while

  return result;
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

