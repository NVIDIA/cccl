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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/mismatch.h>
#include <thrust/system/detail/generic/tag.h>

#include <cuda/std/functional>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
_CCCL_HOST_DEVICE bool
equal(thrust::execution_policy<DerivedPolicy>& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  return thrust::equal(exec, first1, last1, first2, ::cuda::std::equal_to<>());
}

// the == below could be a __host__ function in the case of std::vector::iterator::operator==
// we make this exception for equal and use nv_exec_check_disable because it is used in vector's implementation
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE bool
equal(thrust::execution_policy<DerivedPolicy>& exec,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      BinaryPredicate binary_pred)
{
  return thrust::mismatch(exec, first1, last1, first2, binary_pred).first == last1;
}

} // namespace system::detail::generic
THRUST_NAMESPACE_END
