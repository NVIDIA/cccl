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
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

template <typename System, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy(const thrust::detail::execution_policy_base<System>& system,
     InputIterator first,
     InputIterator last,
     OutputIterator result);

template <typename System, typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy_n(const thrust::detail::execution_policy_base<System>& system, InputIterator first, Size n, OutputIterator result);

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result);

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(InputIterator first, Size n, OutputIterator result);

namespace detail
{
template <typename FromSystem, typename ToSystem, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator two_system_copy(
  const thrust::execution_policy<FromSystem>& from_system,
  const thrust::execution_policy<ToSystem>& two_system,
  InputIterator first,
  InputIterator last,
  OutputIterator result);

template <typename FromSystem, typename ToSystem, typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator two_system_copy_n(
  const thrust::execution_policy<FromSystem>& from_system,
  const thrust::execution_policy<ToSystem>& two_system,
  InputIterator first,
  Size n,
  OutputIterator result);
} // namespace detail

THRUST_NAMESPACE_END

#include <thrust/detail/copy.inl>
