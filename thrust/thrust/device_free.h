// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief Deallocates storage allocated by \p device_malloc.
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
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_free deallocates memory allocated by the function \p device_malloc.
 *
 *  \param ptr A \p device_ptr pointing to memory to be deallocated.
 *
 *  The following code snippet demonstrates how to use \p device_free to
 *  deallocate memory allocated by \p device_malloc.
 *
 *  \code
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_free.h>
 *  ...
 *  // allocate some integers with device_malloc
 *  const int N = 100;
 *  thrust::device_ptr<int> int_array = thrust::device_malloc<int>(N);
 *
 *  // manipulate integers
 *  ...
 *
 *  // deallocate with device_free
 *  thrust::device_free(int_array);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_malloc
 */
inline void device_free(thrust::device_ptr<void> ptr)
{
  using thrust::system::detail::generic::select_system;

  using system = thrust::iterator_system<thrust::device_ptr<void>>::type;

  // XXX lower to select_system(system) here
  system s;

  thrust::free(s, ptr);
}

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
