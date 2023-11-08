/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <tuple>

THRUST_NAMESPACE_BEGIN


// define null_type for backwards compatability
struct null_type {};

__host__ __device__ inline
bool operator==(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator<(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator>(const null_type&, const null_type&) { return false; }

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T>
using tuple_element = ::cuda::std::tuple_element<N, T>;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T>
using tuple_size = ::cuda::std::tuple_size<T>;

/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class... T>
using tuple = ::cuda::std::tuple<T...>;

using ::cuda::std::get;
using ::cuda::std::make_tuple;
using ::cuda::std::tie;

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<>
struct tuple_size<tuple<THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<>> {};

template<class T0>
struct tuple_size<tuple<T0, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0>> {};

template<class T0, class T1>
struct tuple_size<tuple<T0, T1, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1>> {};

template<class T0, class T1, class T2>
struct tuple_size<tuple<T0, T1, T2, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2>> {};

template<class T0, class T1, class T2, class T3>
struct tuple_size<tuple<T0, T1, T2, T3, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3>> {};

template<class T0, class T1, class T2, class T3, class T4>
struct tuple_size<tuple<T0, T1, T2, T3, T4, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4>> {};

template<class T0, class T1, class T2, class T3, class T4, class T5>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5>> {};

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6>> {};

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7>> {};

_LIBCUDACXX_END_NAMESPACE_STD
