// SPDX-FileCopyrightText: Copyright (c) 2008-2024, NVIDIA Corporation. All rights reserved.
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

#include <thrust/detail/execution_policy.h>
#include <thrust/detail/pointer.h>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<void, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n);

template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<T, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n);

template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void free(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Pointer ptr);

THRUST_NAMESPACE_END
