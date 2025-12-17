/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/mismatch.h>
#include <thrust/system/detail/generic/equal.h>

#include <cuda/std/__functional/operations.h>

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
