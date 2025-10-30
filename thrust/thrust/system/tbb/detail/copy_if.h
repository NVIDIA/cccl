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
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator
copy_if(tag, InputIterator1 first, InputIterator1 last, InputIterator2 stencil, OutputIterator result, Predicate pred);

} // namespace system::tbb::detail
THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/copy_if.inl>
