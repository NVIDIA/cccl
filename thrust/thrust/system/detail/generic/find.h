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
template <typename DerivedPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE InputIterator
find(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, const T& value);

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator
find_if(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred);

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator
find_if_not(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/find.inl>
