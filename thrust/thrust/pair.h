// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file pair.h
 *  \brief A type encapsulating a heterogeneous pair of elements
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

#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

// Documented in tuple.h — only declared here for compatibility with pair users.
using ::cuda::std::tuple_element;
using ::cuda::std::tuple_size;

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <class T, class U>
using pair = ::cuda::std::pair<T, U>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using ::cuda::std::pair;
#endif // _CCCL_DOXYGEN_INVOKED

using ::cuda::std::get;
using ::cuda::std::make_pair;

/*! \} // pair
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
