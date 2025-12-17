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

/*! \file device_new.h
 *  \brief Constructs new elements in device memory
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

#include <thrust/detail/allocator/value_initialize_range.h>
#include <thrust/device_allocator.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

THRUST_NAMESPACE_BEGIN

/*!
 *  \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_new implements the placement \c new operator for types
 *  resident in device memory. \p device_new calls <tt>T</tt>'s null
 *  constructor on a array of objects in device memory.
 *  No memory is allocated by this function.
 *
 *  \param  p A \p device_ptr to a region of device memory into which
 *          to construct one or many <tt>T</tt>s.
 *  \param  n The number of objects to construct at \p p.
 *  \return p, casted to <tt>T</tt>'s type.
 *
 *  \see device_ptr
 */
template <typename T>
device_ptr<T> device_new(device_ptr<void> p, const size_t n = 1)
{
  auto* dev_ptr = static_cast<T*>(p.get());
  // TODO(bgruber): ideally, we would have an thrust::uninitialized_default_construct. Until then, use vector's
  // infrastructure
  device_allocator<T> alloc; // not needed for allocation, just for construct() called in value_initialize_range()
  detail::value_initialize_range(alloc, dev_ptr, n);
  return device_ptr<T>{dev_ptr};
}

/*! \p device_new implements the placement new operator for types
 *  resident in device memory. \p device_new calls <tt>T</tt>'s copy
 *  constructor on a array of objects in device memory. No memory is
 *  allocated by this function.
 *
 *  \param  p A \p device_ptr to a region of device memory into which to
 *          construct one or many <tt>T</tt>s.
 *  \param exemplar The value from which to copy.
 *  \param  n The number of objects to construct at \p p.
 *  \return p, casted to <tt>T</tt>'s type.
 *
 *  \see device_ptr
 *  \see fill
 */
template <typename T>
device_ptr<T> device_new(device_ptr<void> p, const T& exemplar, const size_t n = 1)
{
  device_ptr<T> result(static_cast<T*>(p.get()));

  // run copy constructors at p here
  thrust::uninitialized_fill(device, result, result + n, exemplar);

  return result;
}

/*! \p device_new implements the new operator for types resident in device memory.
 *  It allocates device memory large enough to hold \p n new objects of type \c T.
 *
 *  \param n The number of objects to allocate. Defaults to \c 1.
 *  \return A \p device_ptr to the newly allocated region of device memory.
 */
template <typename T>
device_ptr<T> device_new(const size_t n = 1)
{
  // call placement new version of device_new
  return device_new<T>(thrust::device_malloc<T>(n));
}

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
