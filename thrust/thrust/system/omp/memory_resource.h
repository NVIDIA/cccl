/*
 *  Copyright 2018-2020 NVIDIA Corporation
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
namespace system
{
namespace omp
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
// FIXME(bgruber): comment below is wrong or alias should be to universal_memory_resource
/*! An alias for \p omp::universal_memory_resource. */
using universal_host_pinned_memory_resource = detail::native_resource;

/*! \}
 */

} // namespace omp
} // namespace system

THRUST_NAMESPACE_END
