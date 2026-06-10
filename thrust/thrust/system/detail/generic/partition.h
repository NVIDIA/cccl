// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file partition.h
 *  \brief Generic implementations of partition functions.
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
template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator stable_partition(
  thrust::execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator stable_partition(
  thrust::execution_policy<ExecutionPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred);

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator
partition(thrust::execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator partition(
  thrust::execution_policy<ExecutionPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> partition_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> partition_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred);

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator partition_point(
  thrust::execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool is_partitioned(
  thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/partition.inl>
