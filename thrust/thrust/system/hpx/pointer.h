// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! \file
//! Managing memory associated with Thrust's HPX system.

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
#include <thrust/system/hpx/detail/execution_policy.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN
namespace system::hpx
{
/*! \p hpx::pointer stores a pointer to an object allocated in memory accessible
 *  by the \p hpx system. This type provides type safety when dispatching
 *  algorithms on ranges resident in \p hpx memory.
 *
 *  \p hpx::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p hpx::pointer can be created with the function \p hpx::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p hpx::pointer may be obtained by either its
 *  <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p hpx::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p hpx::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see hpx::malloc
 *  \see hpx::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<T, thrust::system::hpx::tag, thrust::tagged_reference<T, thrust::system::hpx::tag>>;

/*! \p hpx::universal_pointer stores a pointer to an object allocated in memory
 * accessible by the \p hpx system and host systems.
 *
 *  \p hpx::universal_pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p hpx::universal_pointer can be created with \p hpx::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p hpx::universal_pointer may be obtained
 *  by either its <tt>get</tt> member function or the \p raw_pointer_cast
 *  function.
 *
 *  \note \p hpx::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p hpx::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see hpx::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<T, thrust::system::hpx::tag, typename std::add_lvalue_reference<T>::type>;

/*! \p reference is a wrapped reference to an object stored in memory available
 *  to the \p hpx system. \p reference is the type of the result of
 *  dereferencing a \p hpx::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
using reference = thrust::reference<T, thrust::system::hpx::tag>;
} // namespace system::hpx

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::hpx
 *  \brief \p thrust::hpx is a top-level alias for \p thrust::system::hpx. */
namespace hpx
{
using thrust::system::hpx::pointer;
using thrust::system::hpx::reference;
using thrust::system::hpx::universal_pointer;
} // namespace hpx

THRUST_NAMESPACE_END
