/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file thrust/system/omp/memory.h
 *  \brief Managing memory associated with Thrust's OpenMP system.
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
#include <thrust/system/omp/memory_resource.h>

#include <cuda/std/limits>

#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system::omp
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

/*! Allocates an area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>omp::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<void></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
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

/*! Allocates a typed area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>omp::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<T></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
 *  \see std::malloc
 */
template <typename T>
inline pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::omp::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

/*! Deallocates an area of memory previously allocated by <tt>omp::malloc</tt>.
 *  \param ptr A <tt>omp::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>omp::malloc</tt>.
 *  \see omp::malloc
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

/*! \p omp::allocator is the default allocator used by the \p omp system's
 *  containers such as <tt>omp::vector</tt> if no user-specified allocator is
 *  provided. \p omp::allocator allocates (deallocates) storage with \p
 *  omp::malloc (\p omp::free).
 */
template <typename T>
using allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::omp::memory_resource>;

//! \p omp::universal_allocator allocates memory that can be used by the \p omp system and host systems.
template <typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<T, thrust::system::omp::universal_memory_resource>;

//! \p omp::universal_host_pinned_allocator allocates memory that can be used by the \p omp system and host systems.
template <typename T>
using universal_host_pinned_allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::system::omp::universal_host_pinned_memory_resource>;
} // namespace system::omp

/*! \namespace thrust::omp
 *  \brief \p thrust::omp is a top-level alias for thrust::system::omp.
 */
namespace omp
{
using thrust::system::omp::allocator;
using thrust::system::omp::free;
using thrust::system::omp::malloc;
using thrust::system::omp::universal_allocator;
using thrust::system::omp::universal_host_pinned_allocator;
} // namespace omp

THRUST_NAMESPACE_END
