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
#include <thrust/system/detail/generic/partition.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator
stable_partition(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  // tbb prefers generic::stable_partition to cpp::stable_partition
  return thrust::system::detail::generic::stable_partition(exec, first, last, pred);
} // end stable_partition()

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator stable_partition(
  execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  // tbb prefers generic::stable_partition to cpp::stable_partition
  return thrust::system::detail::generic::stable_partition(exec, first, last, stencil, pred);
} // end stable_partition()

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  // tbb prefers generic::stable_partition_copy to cpp::stable_partition_copy
  return thrust::system::detail::generic::stable_partition_copy(exec, first, last, out_true, out_false, pred);
} // end stable_partition_copy()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  // tbb prefers generic::stable_partition_copy to cpp::stable_partition_copy
  return thrust::system::detail::generic::stable_partition_copy(exec, first, last, stencil, out_true, out_false, pred);
} // end stable_partition_copy()
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END
