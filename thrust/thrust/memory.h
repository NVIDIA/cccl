/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/memory.h
 *  \brief Abstractions for Thrust's memory model.
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
#include <thrust/detail/pointer.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/temporary_buffer.h>
#include <thrust/detail/type_traits/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

#ifndef _CCCL_DOXYGEN_INVOKED // Doxygen cannot handle both versions

/*! This version of \p malloc allocates untyped uninitialized storage associated with a given system.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The number of bytes of storage to allocate.
 *  \return If allocation succeeds, a pointer to the allocated storage; a null pointer otherwise.
 *          The pointer must be deallocated with \p thrust::free.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publicly derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p malloc to allocate a range of memory
 *  associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate some memory with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<void,thrust::device_space_tag> void_ptr = thrust::malloc(device_sys, N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate void_ptr with thrust::free
 *  thrust::free(device_sys, void_ptr);
 *  \endcode
 *
 *  \see free
 *  \see device_malloc
 */
template <typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<void, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& system, std::size_t n);

#endif // _CCCL_DOXYGEN_INVOKED

/*! This version of \p malloc allocates typed uninitialized storage associated with a given system.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The number of elements of type \c T which the storage should accommodate.
 *  \return If allocation succeeds, a pointer to an allocation large enough to accommodate \c n
 *          elements of type \c T; a null pointer otherwise.
 *          The pointer must be deallocated with \p thrust::free.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publicly derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p malloc to allocate a range of memory
 *  to accommodate integers associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate ptr with thrust::free
 *  thrust::free(device_sys, ptr);
 *  \endcode
 *
 *  \see free
 *  \see device_malloc
 */
template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<T, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& system, std::size_t n);

/*! \p get_temporary_buffer returns a pointer to storage associated with a given Thrust system sufficient to store up to
 *  \p n objects of type \c T. If not enough storage is available to accommodate \p n objects, an implementation may
 * return a smaller buffer. The number of objects the returned buffer can accommodate is also returned.
 *
 *  Thrust uses \p get_temporary_buffer internally when allocating temporary storage required by algorithm
 * implementations.
 *
 *  The storage allocated with \p get_temporary_buffer must be returned to the system with \p return_temporary_buffer.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The requested number of objects of type \c T the storage should accommodate.
 *  \return A pair \c p such that <tt>p.first</tt> is a pointer to the allocated storage and <tt>p.second</tt> is the
 * number of contiguous objects of type \c T that the storage can accommodate. If no storage can be allocated,
 * <tt>p.first</tt> if no storage can be obtained. The storage must be returned to the system using \p
 * return_temporary_buffer.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publicly derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p get_temporary_buffer to allocate a range of memory
 *  to accommodate integers associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::get_temporary_buffer
 *  const int N = 100;
 *
 *  using ptr_and_size_t = cuda::std::pair<
 *    thrust::pointer<int,thrust::device_system_tag>,
 *    std::ptrdiff_t
 *  >;
 *
 *  thrust::device_system_tag device_sys;
 *  ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
 *
 *  // manipulate up to 100 ints
 *  for(int i = 0; i < ptr_and_size.second; ++i)
 *  {
 *    *ptr_and_size.first = i;
 *  }
 *
 *  // deallocate storage with thrust::return_temporary_buffer
 *  thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
 *  \endcode
 *
 *  \see malloc
 *  \see return_temporary_buffer
 */
template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE ::cuda::std::pair<thrust::pointer<T, DerivedPolicy>,
                                    typename thrust::pointer<T, DerivedPolicy>::difference_type>
get_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy>& system,
                     typename thrust::pointer<T, DerivedPolicy>::difference_type n);

/*! \p free deallocates the storage previously allocated by \p thrust::malloc.
 *
 *  \param system The Thrust system with which the storage is associated.
 *  \param ptr A pointer previously returned by \p thrust::malloc. If \p ptr is null, \p free
 *         does nothing.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p ptr shall have been returned by a previous call to <tt>thrust::malloc(system, n)</tt> or
 * <tt>thrust::malloc<T>(system, n)</tt> for some type \c T.
 *
 *  The following code snippet demonstrates how to use \p free to deallocate a range of memory
 *  previously allocated with \p thrust::malloc.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
 *
 *  // mainpulate memory
 *  ...
 *
 *  // deallocate ptr with thrust::free
 *  thrust::free(device_sys, ptr);
 *  \endcode
 */
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void free(const thrust::detail::execution_policy_base<DerivedPolicy>& system, Pointer ptr);

/*! \p return_temporary_buffer deallocates storage associated with a given Thrust system previously allocated by \p
 * get_temporary_buffer.
 *
 *  Thrust uses \p return_temporary_buffer internally when deallocating temporary storage required by algorithm
 * implementations.
 *
 *  \param system The Thrust system with which the storage is associated.
 *  \param p A pointer previously returned by \p thrust::get_temporary_buffer. If \p ptr is null, \p
 * return_temporary_buffer does nothing.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p p shall have been previously allocated by \p thrust::get_temporary_buffer.
 *
 *  The following code snippet demonstrates how to use \p return_temporary_buffer to deallocate a range of memory
 *  previously allocated by \p get_temporary_buffer.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::get_temporary_buffer
 *  const int N = 100;
 *
 *  using ptr_and_size_t = cuda::std::pair<
 *    thrust::pointer<int,thrust::device_system_tag>,
 *    std::ptrdiff_t
 *  >;
 *
 *  thrust::device_system_tag device_sys;
 *  ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
 *
 *  // manipulate up to 100 ints
 *  for(int i = 0; i < ptr_and_size.second; ++i)
 *  {
 *    *ptr_and_size.first = i;
 *  }
 *
 *  // deallocate storage with thrust::return_temporary_buffer
 *  thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
 *  \endcode
 *
 *  \see free
 *  \see get_temporary_buffer
 */
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void return_temporary_buffer(
  const thrust::detail::execution_policy_base<DerivedPolicy>& system, Pointer p, std::ptrdiff_t n);

/*! \p raw_pointer_cast creates a "raw" pointer from a pointer-like type,
 *  simply returning the wrapped pointer, should it exist.
 *
 *  \param ptr The pointer of interest.
 *  \return <tt>ptr.get()</tt>, if the expression is well formed; <tt>ptr</tt>, otherwise.
 *  \see raw_reference_cast
 */
template <typename Pointer>
_CCCL_HOST_DEVICE typename thrust::detail::pointer_traits<Pointer>::raw_pointer raw_pointer_cast(Pointer ptr);

/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return The raw reference obtained by dereferencing the result of \p raw_pointer_cast applied to the address of ref.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template <typename T>
_CCCL_HOST_DEVICE typename detail::raw_reference<T>::type raw_reference_cast(T& ref);

/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return The raw reference obtained by dereferencing the result of \p raw_pointer_cast applied to the address of ref.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template <typename T>
_CCCL_HOST_DEVICE typename detail::raw_reference<const T>::type raw_reference_cast(const T& ref);

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
