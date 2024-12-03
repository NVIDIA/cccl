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

#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::tuple_element;
#endif // _CCCL_DOXYGEN_INVOKED

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::tuple_size;
#endif // _CCCL_DOXYGEN_INVOKED

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
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <class T, class U>
using pair = _CUDA_VSTD::pair<T, U>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::pair;
#endif // _CCCL_DOXYGEN_INVOKED

using _CUDA_VSTD::get;
using _CUDA_VSTD::make_pair;

/*! \endcond
 */

/*! \} // pair
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
