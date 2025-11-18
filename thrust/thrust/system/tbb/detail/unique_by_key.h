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
#include <thrust/system/detail/generic/unique_by_key.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first,
  BinaryPredicate binary_pred)
{
  // tbb prefers generic::unique_by_key to cpp::unique_by_key
  return thrust::system::detail::generic::unique_by_key(exec, keys_first, keys_last, values_first, binary_pred);
} // end unique_by_key()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  execution_policy<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred)
{
  // tbb prefers generic::unique_by_key_copy to cpp::unique_by_key_copy
  return thrust::system::detail::generic::unique_by_key_copy(
    exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end unique_by_key_copy()
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END
