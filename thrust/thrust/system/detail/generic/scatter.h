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
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
scatter(thrust::execution_policy<DerivedPolicy>& exec,
        InputIterator1 first,
        InputIterator1 last,
        InputIterator2 map,
        RandomAccessIterator output);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
_CCCL_HOST_DEVICE void scatter_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
_CCCL_HOST_DEVICE void scatter_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output,
  Predicate pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scatter.inl>
