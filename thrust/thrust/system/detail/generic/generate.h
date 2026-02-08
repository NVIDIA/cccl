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

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename ForwardIterator, typename Generator>
_CCCL_HOST_DEVICE void
generate(thrust::execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Generator gen);

template <typename ExecutionPolicy, typename OutputIterator, typename Size, typename Generator>
_CCCL_HOST_DEVICE OutputIterator
generate_n(thrust::execution_policy<ExecutionPolicy>& exec, OutputIterator first, Size n, Generator gen);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/generate.inl>
