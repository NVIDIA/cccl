/*
 *  Copyright 2018-2025 NVIDIA Corporation
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

/*! \file hpx/memory_resource.h
 *  \brief Memory resources for the HPX system.
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
#include <thrust/system/hpx/pointer.h>

#include <hpx/parallel/algorithms/for_loop.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{

//! \cond
namespace detail
{

class topology_resource final : public mr::memory_resource<>
{
public:
  void* do_allocate(std::size_t bytes, std::size_t /*alignment*/ = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    std::byte* ptr = static_cast<std::byte*>(::hpx::threads::create_topology().allocate(bytes));

    // touch first byte of every page
    const auto page_size = ::hpx::threads::get_memory_page_size();
    ::hpx::experimental::for_loop_strided(::hpx::execution::par, ptr, ptr + bytes, page_size, [](std::byte* it) {
      *it = {};
    });

    return ptr;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t /*alignment*/ = THRUST_MR_DEFAULT_ALIGNMENT) override
  {
    return ::hpx::threads::create_topology().deallocate(p, bytes);
  }
};

using native_resource = thrust::mr::fancy_pointer_resource<topology_resource, thrust::hpx::pointer<void>>;

using universal_native_resource =
  thrust::mr::fancy_pointer_resource<thrust::mr::new_delete_resource, thrust::hpx::universal_pointer<void>>;
} // namespace detail
//! \endcond

/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! The memory resource for the HPX system. Uses \p mr::new_delete_resource and
 *  tags it with \p hpx::pointer.
 */
using memory_resource = detail::native_resource;
/*! The unified memory resource for the HPX system. Uses
 *  \p mr::new_delete_resource and tags it with \p hpx::universal_pointer.
 */
using universal_memory_resource = detail::universal_native_resource;

/*! An alias for \p hpx::universal_memory_resource. */
using universal_host_pinned_memory_resource = universal_memory_resource;

/*! \} // memory_resources
 */

} // namespace hpx
} // namespace system

THRUST_NAMESPACE_END
