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

template <typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
count(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
      InputIterator first,
      InputIterator last,
      const EqualityComparable& value);

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
count_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         InputIterator first,
         InputIterator last,
         Predicate pred);

template <typename InputIterator, typename EqualityComparable>
thrust::detail::it_difference_t<InputIterator>
count(InputIterator first, InputIterator last, const EqualityComparable& value);

template <typename InputIterator, typename Predicate>
thrust::detail::it_difference_t<InputIterator> count_if(InputIterator first, InputIterator last, Predicate pred);

THRUST_NAMESPACE_END

#include <thrust/detail/count.inl>
