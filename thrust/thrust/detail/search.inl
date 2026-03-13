// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include generic implementation
#include <thrust/system/detail/generic/search.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last)
{
  using thrust::system::detail::generic::search;
  return search(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, s_first, s_last);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last,
  BinaryPredicate pred)
{
  using thrust::system::detail::generic::search;
  return search(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, s_first, s_last, pred);
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1 search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<ForwardIterator1>::type;
  using System2 = typename thrust::iterator_system<ForwardIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::search(select_system(system1, system2), first, last, s_first, s_last);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
ForwardIterator1 search(
  ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<ForwardIterator1>::type;
  using System2 = typename thrust::iterator_system<ForwardIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::search(select_system(system1, system2), first, last, s_first, s_last, pred);
}

THRUST_NAMESPACE_END
