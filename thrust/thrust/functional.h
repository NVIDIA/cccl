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

/*! \file functional.h
 *  \brief Function objects and tools for manipulating them
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

#include <thrust/detail/functional/actor.h>

#include <cuda/functional>
#include <cuda/std/functional>

#include <functional>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(func, impl)                                                      \
  template <>                                                                                                      \
  struct func<void>                                                                                                \
  {                                                                                                                \
    using is_transparent = void;                                                                                   \
    _CCCL_EXEC_CHECK_DISABLE                                                                                       \
    template <typename T1, typename T2>                                                                            \
    _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const noexcept(noexcept(impl)) -> decltype(impl) \
    {                                                                                                              \
      return impl;                                                                                                 \
    }                                                                                                              \
  }

/*! \p plus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>plus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x+y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x+y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>plus</tt> to sum two
 *  device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::plus<float>());
 *  // V3 is now {76, 77, 78, ..., 1075}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/plus
 */
using ::cuda::std::plus;

/*! \p minus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>minus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x-y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x-y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>minus</tt> to subtract
 *  a device_vector of \c floats from another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::minus<float>());
 *  // V3 is now {-74, -73, -72, ..., 925}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/minus
 */
using ::cuda::std::minus;

/*! \p multiplies is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>multiplies<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x*y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x*y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>multiplies</tt> to multiply
 *  two device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::multiplies<float>());
 *  // V3 is now {75, 150, 225, ..., 75000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/multiplies
 */
using ::cuda::std::multiplies;

/*! \p divides is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>divides<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x/y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x/y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>divides</tt> to divide
 *  one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::divides<float>());
 *  // V3 is now {1/75, 2/75, 3/75, ..., 1000/75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/divides
 */
using ::cuda::std::divides;

/*! \p modulus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>modulus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x \% y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x \% y</tt> must be defined and must have a return
 * type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>modulus</tt> to take
 *  the modulus of one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::modulus<int>());
 *  // V3 is now {1%75, 2%75, 3%75, ..., 1000%75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/modulus
 */
using ::cuda::std::modulus;

/*! \p negate is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>negate<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>-x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>-x</tt> must be defined and must have a return type that is
 * convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>negate</tt> to negate
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::negate<float>());
 *  // V2 is now {-1, -2, -3, ..., -1000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/negate
 */
using ::cuda::std::negate;

/*! \p square is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>square<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x*x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x*x</tt> must be defined and must have a return type that is
 * convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>square</tt> to square
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::square<float>());
 *  // V2 is now {1, 4, 9, ..., 1000000}
 *  \endcode
 */
template <typename T = void>
struct square
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T operator()(const T& x) const
  {
    return x * x;
  }
};

template <>
struct square<void>
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  _CCCL_HOST_DEVICE constexpr T operator()(const T& x) const noexcept(noexcept(x * x))
  {
    return x * x;
  }
};

/*! \}
 */

/*! \addtogroup comparison_operations Comparison Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x == y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/equal_to
 */
using ::cuda::std::equal_to;

/*! \p not_equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>not_equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x != y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/not_equal_to
 */
using ::cuda::std::not_equal_to;

/*! \p greater is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x > y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater
 */
using ::cuda::std::greater;

/*! \p less is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x < y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less
 */
using ::cuda::std::less;

/*! \p greater_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x >= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater_equal
 */
using ::cuda::std::greater_equal;

/*! \p less_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x <= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less_equal
 */
using ::cuda::std::less_equal;

/*! \}
 */

/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p logical_and is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_and<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if both \c x and \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_and
 */
using ::cuda::std::logical_and;

/*! \p logical_or is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_or<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if either \c x or \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_or
 */
using ::cuda::std::logical_or;

/*! \p logical_not is a function object. Specifically, it is an Adaptable Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_not<T></tt> and \c x is an object of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x)</tt> returns \c true
 *  if and only if \c x is \c false.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  The following code snippet demonstrates how to use \p logical_not to transform
 *  a device_vector of \c bools into its logical complement.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<bool> V;
 *  ...
 *  thrust::transform(V.begin(), V.end(), V.begin(), thrust::logical_not<bool>());
 *  // The elements of V are now the logical complement of what they were prior
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_not
 */
using ::cuda::std::logical_not;

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p bit_and is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x&y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x&y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_and</tt> to take
 *  the bitwise AND of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_and<int>());
 *  // V3 is now {1&13, 2&13, 3&13, ..., 1000%13}
 *  \endcode
 */
using ::cuda::std::bit_and;

/*! \p bit_or is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x|y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x|y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_or</tt> to take
 *  the bitwise OR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_or<int>());
 *  // V3 is now {1|13, 2|13, 3|13, ..., 1000|13}
 *  \endcode
 */
using ::cuda::std::bit_or;

/*! \p bit_xor is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x^y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x^y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_xor</tt> to take
 *  the bitwise XOR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_xor<int>());
 *  // V3 is now {1^13, 2^13, 3^13, ..., 1000^13}
 *  \endcode
 */
using ::cuda::std::bit_xor;

/*! \}
 */

/*! \addtogroup generalized_identity_operations Generalized Identity Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p identity is a Unary Function that represents the identity function: it takes
 *  a single argument \c x, and returns \c x.
 *
 *  \tparam T No requirements on \p T.
 *
 *  The following code snippet demonstrates that \p identity returns its
 *  argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x = 137;
 *  thrust::identity<int> id;
 *  assert(x == id(x));
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/identity
 */
// TODO(bgruber): this version can also act as a functor casting to T making it not equivalent to ::cuda::std::identity
template <typename T = void>
struct identity
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T& operator()(const T& x) const
  {
    return x;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T& operator()(T& x) const
  {
    return x;
  }

  // we cannot add an overload for `const T&&` because then calling e.g. `thrust::identity<int>{}(3.14);` is ambiguous
  // on MSVC

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T&& operator()(T&& x) const
  {
    return _CUDA_VSTD::move(x);
  }
};

template <>
struct identity<void> : ::cuda::std::__identity
{};

/*! \p maximum is a function object that takes two arguments and returns the greater
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>maximum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x > y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  The following code snippet demonstrates that \p maximum returns its
 *  greater argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::maximum<int> mx;
 *  assert(x == mx(x,y));
 *  \endcode
 *
 *  \see minimum
 *  \see min
 */
using ::cuda::maximum;

/*! \p minimum is a function object that takes two arguments and returns the lesser
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>minimum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x < y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  The following code snippet demonstrates that \p minimum returns its
 *  lesser argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::minimum<int> mn;
 *  assert(y == mn(x,y));
 *  \endcode
 *
 *  \see maximum
 *  \see max
 */
using ::cuda::minimum;

/*! \p project1st is a function object that takes two arguments and returns
 *  its first argument; the second argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project1st<int> pj1;
 *  assert(x == pj1(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project2nd
 */
template <typename T1 = void, typename T2 = void>
struct project1st
{
  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  _CCCL_HOST_DEVICE constexpr const T1& operator()(const T1& lhs, const T2& /*rhs*/) const
  {
    return lhs;
  }
};

template <>
struct project1st<void, void>
{
  using is_transparent = void;
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&&) const noexcept(noexcept(THRUST_FWD(t1)))
    -> decltype(THRUST_FWD(t1))
  {
    return THRUST_FWD(t1);
  }
};

/*! \p project2nd is a function object that takes two arguments and returns
 *  its second argument; the first argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project2nd<int> pj2;
 *  assert(y == pj2(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project1st
 */
template <typename T1 = void, typename T2 = void>
struct project2nd
{
  /*! Function call operator. The return value is <tt>rhs</tt>.
   */
  _CCCL_HOST_DEVICE constexpr const T2& operator()(const T1& /*lhs*/, const T2& rhs) const
  {
    return rhs;
  }
}; // end project2nd

template <>
struct project2nd<void, void>
{
  using is_transparent = void;
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&&, T2&& t2) const noexcept(noexcept(THRUST_FWD(t2)))
    -> decltype(THRUST_FWD(t2))
  {
    return THRUST_FWD(t2);
  }
};

/*! \}
 */

// odds and ends

/*! \addtogroup function_object_adaptors
 *  \{
 */

using ::cuda::std::not_fn;

/*! \}
 */

/*! \addtogroup placeholder_objects Placeholder Objects
 *  \ingroup function_objects
 *  \{
 */

/*! \namespace thrust::placeholders
 *  \brief Facilities for constructing simple functions inline.
 *
 *  Objects in the \p thrust::placeholders namespace may be used to create simple arithmetic functions inline
 *  in an algorithm invocation. Combining placeholders such as \p _1 and \p _2 with arithmetic operations such as \c +
 *  creates an unnamed function object which applies the operation to their arguments.
 *
 *  The type of placeholder objects is implementation-defined.
 *
 *  The following code snippet demonstrates how to use the placeholders \p _1 and \p _2 with \p thrust::transform
 *  to implement the SAXPY computation:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<float> x(4), y(4);
 *    x[0] = 1;
 *    x[1] = 2;
 *    x[2] = 3;
 *    x[3] = 4;
 *
 *    y[0] = 1;
 *    y[1] = 1;
 *    y[2] = 1;
 *    y[3] = 1;
 *
 *    float a = 2.0f;
 *
 *    using namespace thrust::placeholders;
 *
 *    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
 *      a * _1 + _2
 *    );
 *
 *    // y is now {3, 5, 7, 9}
 *  }
 *  \endcode
 */
namespace placeholders
{

/*! \p thrust::placeholders::_1 is the placeholder for the first function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<0>::type _1;

/*! \p thrust::placeholders::_2 is the placeholder for the second function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<1>::type _2;

/*! \p thrust::placeholders::_3 is the placeholder for the third function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<2>::type _3;

/*! \p thrust::placeholders::_4 is the placeholder for the fourth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<3>::type _4;

/*! \p thrust::placeholders::_5 is the placeholder for the fifth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<4>::type _5;

/*! \p thrust::placeholders::_6 is the placeholder for the sixth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<5>::type _6;

/*! \p thrust::placeholders::_7 is the placeholder for the seventh function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<6>::type _7;

/*! \p thrust::placeholders::_8 is the placeholder for the eighth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<7>::type _8;

/*! \p thrust::placeholders::_9 is the placeholder for the ninth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<8>::type _9;

/*! \p thrust::placeholders::_10 is the placeholder for the tenth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<9>::type _10;

} // namespace placeholders

/*! \} // placeholder_objects
 */

#undef THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION

THRUST_NAMESPACE_END

#include <thrust/detail/functional/operators.h>
#include <thrust/detail/type_traits/is_commutative.h>
