// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/system/detail/generic/remove.h>
#include <thrust/system/omp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator
remove_if(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::system::detail::generic::remove_if(exec, first, last, pred);
}

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator remove_if(
  execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::system::detail::generic::remove_if(exec, first, last, stencil, pred);
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::system::detail::generic::remove_copy_if(exec, first, last, result, pred);
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::system::detail::generic::remove_copy_if(exec, first, last, stencil, result, pred);
}

} // end namespace system::omp::detail
THRUST_NAMESPACE_END
