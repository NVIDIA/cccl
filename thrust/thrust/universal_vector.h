// SPDX-FileCopyrightText: Copyright (c) 2008-2020, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief A dynamically-sizable array of elements which resides in memory
 *         accessible to both hosts and devices.
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

#include <thrust/universal_allocator.h>

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/vector.h>
#  include <thrust/system/cuda/vector.h>
#  include <thrust/system/omp/vector.h>
#  include <thrust/system/tbb/vector.h>
#endif

// #include the device system's vector header
#define __THRUST_DEVICE_SYSTEM_VECTOR_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/vector.h>
#include __THRUST_DEVICE_SYSTEM_VECTOR_HEADER
#undef __THRUST_DEVICE_SYSTEM_VECTOR_HEADER

THRUST_NAMESPACE_BEGIN

/*! \addtogroup containers Containers
 *  \{
 */

/*! A \p universal_vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p universal_vector may vary dynamically; memory management is
 *  automatic. The memory associated with a \p universal_vector resides in memory
 *  accessible to hosts and devices.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p universal_vector.
 *  \see device_vector
 *  \see universal_host_pinned_vector
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
template <typename T, typename Allocator = universal_allocator<T>>
using universal_vector = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::universal_vector<T, Allocator>;

//! Like \ref universal_vector but uses pinned memory when the system supports it.
//! \see device_vector
//! \see universal_vector
template <typename T>
using universal_host_pinned_vector = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::universal_host_pinned_vector<T>;

/*! \} // containers
 */

THRUST_NAMESPACE_END
