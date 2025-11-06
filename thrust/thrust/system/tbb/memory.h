// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/system/tbb/memory.h
 *  \brief Managing memory associated with Thrust's TBB system.
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
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/tbb/memory_resource.h>

#include <cuda/std/limits>

#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system::tbb
{
namespace detail
{
// XXX circular #inclusion problems cause the compiler to believe that cpp::malloc
//     is not defined
//     WAR the problem by using adl to call cpp::malloc, which requires it to depend
//     on a template parameter
template <typename Tag>
pointer<void> malloc_workaround(Tag t, std::size_t n)
{
  return pointer<void>(malloc(t, n));
} // end malloc_workaround()

// XXX circular #inclusion problems cause the compiler to believe that cpp::free
//     is not defined
//     WAR the problem by using adl to call cpp::free, which requires it to depend
//     on a template parameter
template <typename Tag>
void free_workaround(Tag t, pointer<void> ptr)
{
  free(t, ptr.get());
} // end free_workaround()
} // namespace detail

/*! Allocates an area of memory available to Thrust's <tt>tbb</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>tbb::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>tbb::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>tbb::pointer<void></tt> returned by this function must be
 *        deallocated with \p tbb::free.
 *  \see tbb::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // return pointer<void>(thrust::system::cpp::malloc(n))
  //
  return detail::malloc_workaround(cpp::tag(), n);
} // end malloc()

/*! Allocates a typed area of memory available to Thrust's <tt>tbb</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>tbb::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>tbb::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>tbb::pointer<T></tt> returned by this function must be
 *        deallocated with \p tbb::free.
 *  \see tbb::free
 *  \see std::malloc
 */
template <typename T>
inline pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::tbb::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

/*! Deallocates an area of memory previously allocated by <tt>tbb::malloc</tt>.
 *  \param ptr A <tt>tbb::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>tbb::malloc</tt>.
 *  \see tbb::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // thrust::system::cpp::free(ptr)
  //
  detail::free_workaround(cpp::tag(), ptr);
} // end free()

/*! \p tbb::allocator is the default allocator used by the \p tbb system's
 *  containers such as <tt>tbb::vector</tt> if no user-specified allocator is
 *  provided. \p tbb::allocator allocates (deallocates) storage with \p
 *  tbb::malloc (\p tbb::free).
 */
template <typename T>
using allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::tbb::memory_resource>;

//! \p tbb::universal_allocator allocates memory that can be used by the \p tbb system and host systems.
template <typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::tbb::universal_memory_resource>;

//! \p tbb::universal_host_pinned_allocator allocates memory that can be used by the \p tbb system and host systems.
template <typename T>
using universal_host_pinned_allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::system::tbb::universal_host_pinned_memory_resource>;
} // namespace system::tbb

/*! \namespace thrust::tbb
 *  \brief \p thrust::tbb is a top-level alias for thrust::system::tbb.
 */
namespace tbb
{
using thrust::system::tbb::allocator;
using thrust::system::tbb::free;
using thrust::system::tbb::malloc;
using thrust::system::tbb::universal_allocator;
using thrust::system::tbb::universal_host_pinned_allocator;
} // namespace tbb

THRUST_NAMESPACE_END
