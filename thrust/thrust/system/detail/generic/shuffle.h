// SPDX-FileCopyrightText: Copyright (c) 2008-2020, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file shuffle.h
 *  \brief Generic implementations of shuffle functions.
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

#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
_CCCL_HOST_DEVICE void
shuffle(thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first, RandomIterator last, URBG&& g);

template <typename ExecutionPolicy, typename RandomIterator, typename OutputIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  RandomIterator first,
  RandomIterator last,
  OutputIterator result,
  URBG&& g);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/shuffle.inl>
