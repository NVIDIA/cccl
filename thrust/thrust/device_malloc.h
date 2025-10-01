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

/*! \file
 *  \brief Allocates storage in device memory.
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

#include <thrust/detail/malloc_and_free.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/cstddef>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

//! Allocates sequential device storage for new objects of a given type, or raw bytes if the type is \p void.
//!
//! \param n The number of objects of type T, or bytes, to allocate sequentially in device memory.
//! \return A \p device_ptr to the newly allocated memory.
//!
//! The following code snippet demonstrates how to use \p device_malloc to
//! allocate a range of device memory.
//!
//! \code
//! #include <thrust/device_malloc.h>
//! #include <thrust/device_free.h>
//! ...
//! // allocate some integers with device_malloc
//! const int N = 100;
//! thrust::device_ptr<int> int_array = thrust::device_malloc<int>(N);
//! thrust::device_ptr<void> raw_byte_array = thrust::device_malloc(N);
//!
//! // manipulate integers and bytes
//! ...
//!
//! // deallocate with device_free
//! thrust::device_free(raw_byte_array);
//! thrust::device_free(int_array);
//! \endcode
//!
//! \see device_ptr
//! \see device_free
template <typename T = void>
device_ptr<T> device_malloc(const std::size_t n)
{
  // using system::detail::generic::select_system;
  // XXX lower to select_system(system) here
  iterator_system_t<device_ptr<void>> s;
  return thrust::device_ptr<T>(thrust::malloc<T>(s, n).get());
}

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
