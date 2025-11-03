// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
#include <thrust/system/detail/generic/tag.h>

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
