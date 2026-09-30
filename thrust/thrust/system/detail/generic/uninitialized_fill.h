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
template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void uninitialized_fill(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& x);

template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
_CCCL_HOST_DEVICE ForwardIterator
uninitialized_fill_n(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, Size n, const T& x);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/uninitialized_fill.inl>
