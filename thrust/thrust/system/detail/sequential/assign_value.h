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

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(sequential::execution_policy<DerivedPolicy>&, Pointer1 dst, Pointer2 src)
{
  *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
} // end assign_value()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
