// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file thrust/system/hpx/memory.h
 *  \brief Managing memory associated with Thrust's HPX system.
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
#include <thrust/detail/type_traits.h>
#include <thrust/memory.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/hpx/memory_resource.h>

#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system::hpx
{
/*! Allocates an area of memory available to Thrust's <tt>hpx</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>hpx::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>hpx::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>hpx::pointer<void></tt> returned by this function must be
 *        deallocated with \p hpx::free.
 *  \see hpx::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>hpx</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>hpx::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>hpx::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>hpx::pointer<T></tt> returned by this function must be
 *        deallocated with \p hpx::free.
 *  \see hpx::free
 *  \see std::malloc
 */
template <typename T>
inline pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>hpx::malloc</tt>.
 *  \param ptr A <tt>hpx::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>hpx::malloc</tt>.
 *  \see hpx::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr);

/*! \p hpx::allocator is the default allocator used by the \p hpx system's
 *  containers such as <tt>hpx::vector</tt> if no user-specified allocator is
 *  provided. \p hpx::allocator allocates (deallocates) storage with \p
 *  hpx::malloc (\p hpx::free).
 */
template <typename T>
using allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::hpx::memory_resource>;

//! \p hpx::universal_allocator allocates memory that can be used by the \p hpx system and host systems.
template <typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::hpx::universal_memory_resource>;

//! \p hpx::universal_host_pinned_allocator allocates memory that can be used by the \p hpx system and host systems.
template <typename T>
using universal_host_pinned_allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::system::hpx::universal_host_pinned_memory_resource>;
} // namespace system::hpx

/*! \namespace thrust::hpx
 *  \brief \p thrust::hpx is a top-level alias for thrust::system::hpx.
 */
namespace hpx
{
using thrust::system::hpx::allocator;
using thrust::system::hpx::free;
using thrust::system::hpx::malloc;
using thrust::system::hpx::universal_allocator;
using thrust::system::hpx::universal_host_pinned_allocator;
} // namespace hpx

THRUST_NAMESPACE_END

#include <thrust/system/hpx/detail/memory.inl>
