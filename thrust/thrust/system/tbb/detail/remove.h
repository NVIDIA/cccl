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
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator
remove_if(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred);

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator remove_if(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred);

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred);

} // end namespace system::tbb::detail
THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/remove.inl>
