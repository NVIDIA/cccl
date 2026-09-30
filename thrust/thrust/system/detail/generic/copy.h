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
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result);

template <typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy_n(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, Size n, OutputIterator result);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/copy.inl>
