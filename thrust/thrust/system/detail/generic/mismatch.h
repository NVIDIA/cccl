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
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<InputIterator1, InputIterator2> mismatch(
  thrust::execution_policy<DerivedPolicy>& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2);

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<InputIterator1, InputIterator2> mismatch(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  BinaryPredicate pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/mismatch.inl>
