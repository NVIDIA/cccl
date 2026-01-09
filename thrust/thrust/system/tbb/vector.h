// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/system/tbb/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's TBB system.
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
#include <thrust/detail/vector_base.h>
#include <thrust/system/tbb/memory.h>

#include <vector>

THRUST_NAMESPACE_BEGIN
namespace system::tbb
{
/*! \p tbb::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p tbb::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p tbb::vector reside in memory
 *  accessible by the \p tbb system.
 *
 *  \tparam T The element type of the \p tbb::vector.
 *  \tparam Allocator The allocator type of the \p tbb::vector.
 *          Defaults to \p tbb::allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p tbb::vector.
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::tbb::allocator<T>>
using vector = thrust::detail::vector_base<T, Allocator>;

/*! \p tbb::universal_vector is a container that supports random access to
 *  elements, constant time removal of elements at the end, and linear time
 *  insertion and removal of elements at the beginning or in the middle. The
 *  number of elements in a \p tbb::universal_vector may vary dynamically;
 *  memory management is automatic. The elements contained in a
 *  \p tbb::universal_vector reside in memory accessible by the \p tbb system
 *  and host systems.
 *
 *  \tparam T The element type of the \p tbb::universal_vector.
 *  \tparam Allocator The allocator type of the \p tbb::universal_vector.
 *          Defaults to \p tbb::universal_allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p tbb::universal_vector
 *  \see device_vector
 *  \see universal_host_pinned_vector
 */
template <typename T, typename Allocator = thrust::system::tbb::universal_allocator<T>>
using universal_vector = thrust::detail::vector_base<T, Allocator>;

//! Like \ref universal_vector but uses pinned memory when the system supports it.
//! \see device_vector
//! \see universal_vector
template <typename T>
using universal_host_pinned_vector = thrust::detail::vector_base<T, universal_host_pinned_allocator<T>>;
} // namespace system::tbb

namespace tbb
{
using thrust::system::tbb::universal_vector;
using thrust::system::tbb::vector;
} // namespace tbb

THRUST_NAMESPACE_END
