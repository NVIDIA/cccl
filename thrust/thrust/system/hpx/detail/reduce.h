/*
 *  Copyright 2008-2025 NVIDIA Corporation
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
 *  \brief HPX implementation of reduce.
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
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>

#include <hpx/parallel/algorithms/reduce.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator, typename OutputType, typename BinaryFunction>
OutputType reduce(execution_policy<DerivedPolicy>& exec,
                  InputIterator first,
                  InputIterator last,
                  OutputType init,
                  BinaryFunction binary_op)
{
  // wrap binary_op
  wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
      return ::hpx::reduce(
        hpx::detail::to_hpx_execution_policy(exec),
        ::thrust::try_unwrap_contiguous_iterator(first),
        ::thrust::try_unwrap_contiguous_iterator(last),
        init,
        wrapped_binary_op);
  }
  else
  {
    (void) exec;
    return ::hpx::reduce(first, last, init, wrapped_binary_op);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
