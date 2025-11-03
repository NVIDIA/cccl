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

#include <thrust/detail/internal_functional.h>
#include <thrust/generate.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
_CCCL_HOST_DEVICE OutputIterator
fill_n(thrust::execution_policy<DerivedPolicy>& exec, OutputIterator first, Size n, const T& value)
{
  // XXX consider using the placeholder expression _1 = value
  return thrust::generate_n(exec, first, n, thrust::detail::fill_functor<T>{value});
}

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
fill(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& value)
{
  // XXX consider using the placeholder expression _1 = value
  thrust::generate(exec, first, last, thrust::detail::fill_functor<T>{value});
}

} // namespace system::detail::generic
THRUST_NAMESPACE_END
