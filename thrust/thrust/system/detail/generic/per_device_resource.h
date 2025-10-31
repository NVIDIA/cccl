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
#include <thrust/detail/execution_policy.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename MR, typename DerivedPolicy>
_CCCL_HOST MR* get_per_device_resource(thrust::detail::execution_policy_base<DerivedPolicy>&)
{
  return mr::get_global_resource<MR>();
}

} // namespace system::detail::generic
THRUST_NAMESPACE_END
