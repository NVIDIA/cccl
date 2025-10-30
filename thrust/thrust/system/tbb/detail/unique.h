// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thrust/pair.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{

template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred);

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
thrust::detail::it_difference_t<ForwardIterator> unique_count(
  execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred);

} // end namespace system::tbb::detail
THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/unique.inl>
