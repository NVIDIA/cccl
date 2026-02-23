// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief An allocator which allocates storage with \p device_malloc.
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
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/limits>

#include <new>

THRUST_NAMESPACE_BEGIN

// forward declarations to WAR circular #includes
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <typename>
class device_ptr;
template <typename T>
device_ptr<T> device_malloc(const std::size_t n);
#endif // _CCCL_DOXYGEN_INVOKED

/*! \addtogroup allocators Allocators
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_malloc_allocator is a device memory allocator that employs the
 *  \p device_malloc function for allocation.
 *
 *  \p device_malloc_allocator is deprecated in favor of <tt>thrust::mr</tt>
 *      memory resource-based allocators.
 *
 *  \see device_malloc
 *  \see device_ptr
 *  \see device_allocator
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
template <typename T>
class device_malloc_allocator
{
public:
  /*! Type of element allocated, \c T. */
  using value_type = T;

  /*! Pointer to allocation, \c device_ptr<T>. */
  using pointer = device_ptr<T>;

  /*! \c const pointer to allocation, \c device_ptr<const T>. */
  using const_pointer = device_ptr<const T>;

  /*! Reference to allocated element, \c device_reference<T>. */
  using reference = device_reference<T>;

  /*! \c const reference to allocated element, \c device_reference<const T>. */
  using const_reference = device_reference<const T>;

  /*! Type of allocation size, \c std::size_t. */
  using size_type = std::size_t;

  /*! Type of allocation difference, \c pointer::difference_type. */
  using difference_type = typename pointer::difference_type;

  /*! The \p rebind metafunction provides the type of a \p device_malloc_allocator
   *  instantiated with another type.
   *
   *  \tparam U The other type to use for instantiation.
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  template <typename U>
  struct rebind
  {
    /*! The alias \p other gives the type of the rebound \p device_malloc_allocator.
     *
     *  \verbatim embed:rst:leading-asterisk
     *     .. versionadded:: 2.2.0
     *  \endverbatim
     */
    using other = device_malloc_allocator<U>;
  }; // end rebind

  /*! No-argument constructor has no effect. */
  _CCCL_HOST_DEVICE inline device_malloc_allocator() {}

  /*! No-argument destructor has no effect. */
  _CCCL_HOST_DEVICE inline ~device_malloc_allocator() {}

  /*! Copy constructor has no effect. */
  _CCCL_HOST_DEVICE inline device_malloc_allocator(device_malloc_allocator const&) {}

  /*! Constructor from other \p device_malloc_allocator has no effect. */
  template <typename U>
  _CCCL_HOST_DEVICE inline device_malloc_allocator(device_malloc_allocator<U> const&)
  {}

  device_malloc_allocator& operator=(const device_malloc_allocator&) = default;

  /*! Returns the address of an allocated object.
   *  \return <tt>&r</tt>.
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST_DEVICE inline pointer address(reference r)
  {
    return &r;
  }

  /*! Returns the address an allocated object.
   *  \return <tt>&r</tt>.
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST_DEVICE inline const_pointer address(const_reference r)
  {
    return &r;
  }

  /*! Allocates storage for \p cnt objects.
   *  \param cnt The number of objects to allocate.
   *  \return A \p pointer to uninitialized storage for \p cnt objects.
   *  \note Memory allocated by this function must be deallocated with \p deallocate.
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST inline pointer allocate(size_type cnt, const_pointer = const_pointer(static_cast<T*>(0)))
  {
    if (cnt > this->max_size())
    {
      _CCCL_THROW(::std::bad_alloc);
    } // end if

    return pointer(device_malloc<T>(cnt));
  } // end allocate()

  /*! Deallocates storage for objects allocated with \p allocate.
   *  \param p A \p pointer to the storage to deallocate.
   *  \param cnt The size of the previous allocation.
   *  \note Memory deallocated by this function must previously have been
   *        allocated with \p allocate.
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST inline void deallocate(pointer p, [[maybe_unused]] size_type cnt) noexcept
  {
    device_free(p);
  } // end deallocate()

  /*! Returns the largest value \c n for which <tt>allocate(n)</tt> might succeed.
   *  \return The largest value \c n for which <tt>allocate(n)</tt> might succeed.
   */
  inline size_type max_size() const
  {
    return (::cuda::std::numeric_limits<size_type>::max)() / sizeof(T);
  } // end max_size()

  /*! Compares against another \p device_malloc_allocator for equality.
   *  \return \c true
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST_DEVICE inline bool operator==(device_malloc_allocator const&) const
  {
    return true;
  }

  /*! Compares against another \p device_malloc_allocator for inequality.
   *  \return \c false
   *
   *  \verbatim embed:rst:leading-asterisk
   *     .. versionadded:: 2.2.0
   *  \endverbatim
   */
  _CCCL_HOST_DEVICE inline bool operator!=(device_malloc_allocator const& a) const
  {
    return !operator==(a);
  }
}; // end device_malloc_allocator

/*! \} // allocators
 */

THRUST_NAMESPACE_END
