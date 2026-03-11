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
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/cstdlib> // for malloc & free

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
template <typename DerivedPolicy>
inline _CCCL_HOST_DEVICE void* malloc(execution_policy<DerivedPolicy>&, std::size_t n)
{
  return ::cuda::std::malloc(n);
} // end mallc()

template <typename DerivedPolicy, typename Pointer>
inline _CCCL_HOST_DEVICE void free(sequential::execution_policy<DerivedPolicy>&, Pointer ptr)
{
  ::cuda::std::free(thrust::raw_pointer_cast(ptr));
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
