// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief Deletes variables in device memory.
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

#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// define an empty allocator class to use below
struct device_delete_allocator
{};
} // namespace detail

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_delete deletes a \p device_ptr allocated with
 *  \p device_new.
 *
 *  \param ptr The \p device_ptr to delete, assumed to have
 *         been allocated with \p device_new.
 *  \param n The number of objects to destroy at \p ptr. Defaults to \c 1
 *         similar to \p device_new.
 *
 *  \see device_ptr
 *  \see device_new
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
template <typename T>
inline void device_delete(thrust::device_ptr<T> ptr, const size_t n = 1)
{
  // we don't have an allocator, so there is no need to go through thrust::detail::destroy_range
  thrust::for_each_n(device, ptr, n, detail::gozer{});
  thrust::device_free(ptr);
}

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
