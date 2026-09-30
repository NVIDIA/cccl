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
template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result);

// XXX it is an error to call this function; it has no implementation
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result);

// XXX it is an error to call this function; it has no implementation
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  thrust::execution_policy<StrictWeakOrdering>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result);

// XXX it is an error to call this function; it has no implementation
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_union(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result);

// XXX it is an error to call this function; it has no implementation
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_union(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result);

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  thrust::execution_policy<ExecutionPolicy>& system,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/set_operations.inl>
