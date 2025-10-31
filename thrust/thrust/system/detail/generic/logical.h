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
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>
#include <thrust/logical.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
all_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, ::cuda::std::not_fn(pred)) == last;
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
any_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, pred) != last;
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
none_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return !thrust::any_of(exec, first, last, pred);
}

} // namespace system::detail::generic
THRUST_NAMESPACE_END
