// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
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
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/disjoint_tls_pool.h>
#include <thrust/mr/new.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
inline thrust::mr::allocator<
  char,
  thrust::mr::disjoint_unsynchronized_pool_resource<thrust::device_memory_resource, thrust::mr::new_delete_resource>>
single_device_tls_caching_allocator()
{
  return {&thrust::mr::tls_disjoint_pool(thrust::mr::get_global_resource<thrust::device_memory_resource>(),
                                         thrust::mr::get_global_resource<thrust::mr::new_delete_resource>())};
}
} // namespace detail

THRUST_NAMESPACE_END
