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
#include <thrust/system/detail/generic/tag.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  thrust::execution_policy<ExecutionPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first);

template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  thrust::execution_policy<ExecutionPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first,
  BinaryPredicate binary_pred);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/unique_by_key.inl>
