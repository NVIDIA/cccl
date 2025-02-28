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

#include <thrust/for_each.h>

#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename Generator>
struct generate_functor
{
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T&& x)
  {
    ::cuda::std::forward<T>(x) = gen();
  }

  Generator gen;
};

template <typename ExecutionPolicy, typename ForwardIterator, typename Generator>
_CCCL_HOST_DEVICE void
generate(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Generator gen)
{
  thrust::for_each(exec, first, last, generate_functor<Generator>{::cuda::std::move(gen)});
}

template <typename ExecutionPolicy, typename OutputIterator, typename Size, typename Generator>
_CCCL_HOST_DEVICE OutputIterator
generate_n(execution_policy<ExecutionPolicy>& exec, OutputIterator first, Size n, Generator gen)
{
  return thrust::for_each_n(exec, first, n, generate_functor<Generator>{::cuda::std::move(gen)});
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
