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
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator2 swap_ranges(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2);
}
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/swap_ranges.inl>
