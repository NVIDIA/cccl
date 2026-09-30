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
#include <thrust/detail/pointer.h>
#include <thrust/system/detail/generic/tag.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE ::cuda::std::pair<thrust::pointer<T, DerivedPolicy>,
                                    typename thrust::pointer<T, DerivedPolicy>::difference_type>
get_temporary_buffer(thrust::execution_policy<DerivedPolicy>& exec,
                     typename thrust::pointer<T, DerivedPolicy>::difference_type n);

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void
return_temporary_buffer(thrust::execution_policy<DerivedPolicy>& exec, Pointer p, std::ptrdiff_t n);

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void return_temporary_buffer(thrust::execution_policy<DerivedPolicy>& exec, Pointer p);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/temporary_buffer.inl>
