// SPDX-FileCopyrightText: Copyright (c) 2018-2020, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file omp/memory_resource.h
 *  \brief Memory resources for the OpenMP system.
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

#include <thrust/mr/fancy_pointer_resource.h>
#include <thrust/mr/new.h>
#include <thrust/system/omp/pointer.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp
{
//! \cond
namespace detail
{
using native_resource = thrust::mr::fancy_pointer_resource<thrust::mr::new_delete_resource, thrust::omp::pointer<void>>;

using universal_native_resource =
  thrust::mr::fancy_pointer_resource<thrust::mr::new_delete_resource, thrust::omp::universal_pointer<void>>;
} // namespace detail
//! \endcond

/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! The memory resource for the OpenMP system. Uses \p mr::new_delete_resource
 *  and tags it with \p omp::pointer.
 */
using memory_resource = detail::native_resource;
/*! The unified memory resource for the OpenMP system. Uses
 *  \p mr::new_delete_resource and tags it with \p omp::universal_pointer.
 */
using universal_memory_resource = detail::universal_native_resource;

/*! An alias for \p omp::universal_memory_resource. */
using universal_host_pinned_memory_resource = universal_memory_resource;

/*! \}
 */
} // namespace system::omp

THRUST_NAMESPACE_END
