// SPDX-FileCopyrightText: Copyright (c) 2008-2020, NVIDIA Corporation. All rights reserved.
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
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__type_traits/add_lvalue_reference.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb
{
/*! \p tbb::pointer stores a pointer to an object allocated in memory accessible
 *  by the \p tbb system. This type provides type safety when dispatching
 *  algorithms on ranges resident in \p tbb memory.
 *
 *  \p tbb::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p tbb::pointer can be created with the function \p tbb::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p tbb::pointer may be obtained by either its
 *  <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p tbb::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p tbb::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see tbb::malloc
 *  \see tbb::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<T, thrust::system::tbb::tag, thrust::tagged_reference<T, thrust::system::tbb::tag>>;

/*! \p tbb::universal_pointer stores a pointer to an object allocated in memory
 * accessible by the \p tbb system and host systems.
 *
 *  \p tbb::universal_pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p tbb::universal_pointer can be created with \p tbb::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p tbb::universal_pointer may be obtained
 *  by either its <tt>get</tt> member function or the \p raw_pointer_cast
 *  function.
 *
 *  \note \p tbb::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p tbb::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see tbb::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<T, thrust::system::tbb::tag, ::cuda::std::add_lvalue_reference_t<T>>;

template <typename T>
using universal_host_pinned_pointer = universal_pointer<T>;

/*! \p reference is a wrapped reference to an object stored in memory available
 *  to the \p tbb system. \p reference is the type of the result of
 *  dereferencing a \p tbb::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
using reference = thrust::tagged_reference<T, thrust::system::tbb::tag>;
} // namespace system::tbb

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::tbb
 *  \brief \p thrust::tbb is a top-level alias for \p thrust::system::tbb. */
namespace tbb
{
using thrust::system::tbb::pointer;
using thrust::system::tbb::reference;
using thrust::system::tbb::universal_host_pinned_pointer;
using thrust::system::tbb::universal_pointer;
} // namespace tbb

THRUST_NAMESPACE_END
