/*
 *  Copyright 2008-2019 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/complex>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup numerics
 *  \{
 */

/*! \addtogroup complex_numbers Complex Numbers
 *  \{
 */

/*! \p complex is the Thrust equivalent to <tt>std::complex</tt>. It is
 *  functionally identical to it, but can also be used in device code which
 *  <tt>std::complex</tt> currently cannot.
 *
 *  \tparam T The type used to hold the real and imaginary parts. Should be
 *  <tt>float</tt> or <tt>double</tt>. Others types are not supported.
 *
 */
template <class T>
struct complex : public ::cuda::std::complex<T>
{
  using ::cuda::std::complex<T>::complex; // inherit constructors

  /*! Copy constructor for base class with promoted underlying type.
   *
   * This "overwrites" constructor where U == T, alternativley one could use
   * enable_if on dummy parameter.
   */
  template <typename U>
  __host__ __device__ complex(const ::cuda::std::complex<U> &rhs);

  /*! Copy assignment for base class with diffent underlying type. */
  template <typename U>
  __host__ __device__ typename detail::disable_if<detail::is_same<T, U>::value, complex>::type &
  operator=(const ::cuda::std::complex<U> &z);

  /*! Cast this \p complex to a <tt>std::complex</tt>
   */
  __host__ operator std::complex<T>() const { return std::complex<T>(this->real(), this->imag()); }
};

// The following code enables operators between `complex` when the underlying type of the `complex`
// differ. Using enable_if ensures that these operators are only called, in the case
// the types differ. In all other cases the operators from the base class ::cuda::std::complex
// should be called.
// This is done to avoid the construction of temporary objects which must be done in some of
// the operators. Also it is avoided to call the implementation of the base class operator
// in case the operator is trivially implementable, for the same reason.

/* --- Equality Operators --- */

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const complex<T0> &x, const complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p thrust::complex.
 *  \param y The second \p cuda::std::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const complex<T0> &x, const ::cuda::std::complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p cuda::std::complex.
 *  \param y The second \p thrust::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const ::cuda::std::complex<T0> &x, const complex<T1> &y);

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T0>>>::value,
  bool>::type
operator==(const T0 &x, const complex<T1> &y);

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  bool>::type
operator==(const complex<T0> &x, const T1 &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator!=(const complex<T0> &x, const complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The first \p thrust::complex.
 *  \param y The second \p cuda::std::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator!=(const complex<T0> &x, const ::cuda::std::complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The second \p cuda::std::complex.
 *  \param y The first \p thrust::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::enable_if<detail::is_same<T0, T1>::value, bool>::type
operator!=(const ::cuda::std::complex<T0> &x, const complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T0>>>::value,
  bool>::type
operator!=(const T0 &x, const complex<T1> &y);

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  bool>::type
operator!=(const complex<T0> &x, const T1 &y);

/* --- Add Operator --- */

/*! Adds two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator+(const complex<T0> &x, const complex<T1> &y);

/*! Adds a scalar to a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator+(const complex<T0> &x, const T1 &y);

/*! Adds a \p complex number to a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator+(const T0 &x, const complex<T1> &y);

/* --- Substraction Operator --- */

/*! Subtracts two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex (minuend).
 *  \param y The second \p complex (subtrahend).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator-(const complex<T0> &x, const complex<T1> &y);

/*! Subtracts a scalar from a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The \p complex (minuend).
 *  \param y The scalar (subtrahend).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator-(const complex<T0> &x, const T1 &y);

/*! Subtracts a \p complex number from a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar (minuend).
 *  \param y The \p complex (subtrahend).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator-(const T0 &x, const complex<T1> &y);

/* --- Multiplication Operator --- */

/*! Multiplies two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator*(const complex<T0> &x, const complex<T1> &y);

/*! Multiplies a \p complex number by a scalar.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator*(const complex<T0> &x, const T1 &y);

/*! Multiplies a scalar by a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator*(const T0 &x, const complex<T1> &y);

/* --- Division Operator --- */

/*! Divides two \p complex numbers.
 *
 *  The value types of the two \p complex types should be compatible and the
 *  type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The numerator (dividend).
 *  \param y The denomimator (divisor).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator/(const complex<T0> &x, const complex<T1> &y);

/*! Divides a \p complex number by a scalar.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The complex numerator (dividend).
 *  \param y The scalar denomimator (divisor).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator/(const complex<T0> &x, const T1 &y);

/*! Divides a scalar by a \p complex number.
 *
 *  The value type of the \p complex should be compatible with the scalar and
 *  the type of the returned \p complex is the promoted type of the two arguments.
 *
 *  \param x The scalar numerator (dividend).
 *  \param y The complex denomimator (divisor).
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator/(const T0 &x, const complex<T1> &y);

// The using declarations allows imports all necessary functions for thurst::complex.
// However, they also lead to thrust::abs(1.0F) being valid code after include <thurst/complex.h>.
using ::cuda::std::abs;
using ::cuda::std::arg;
using ::cuda::std::conj;
using ::cuda::std::norm;
using ::cuda::std::polar;
using ::cuda::std::proj;

using ::cuda::std::exp;
using ::cuda::std::log;
using ::cuda::std::log10;
using ::cuda::std::pow;
using ::cuda::std::sqrt;

using ::cuda::std::acos;
using ::cuda::std::acosh;
using ::cuda::std::asin;
using ::cuda::std::asinh;
using ::cuda::std::atan;
using ::cuda::std::atanh;
using ::cuda::std::cos;
using ::cuda::std::cosh;
using ::cuda::std::sin;
using ::cuda::std::sinh;
using ::cuda::std::tan;
using ::cuda::std::tanh;

template <typename T>
struct proclaim_trivially_relocatable<complex<T>> : thrust::true_type
{};

/*! Is of `true_type` when \c T is either a thrust::complex, std::complex, or cuda::std::complex.
 */
template <typename T>
struct is_complex : public thrust::false_type
{};
template <typename T>
struct is_complex<complex<T>> : public thrust::true_type
{};
template <typename T>
struct is_complex<::cuda::std::complex<T>> : public thrust::true_type
{};
template <typename T>
struct is_complex<::std::complex<T>> : public thrust::true_type
{};

THRUST_NAMESPACE_END

#include <thrust/detail/complex.inl>

/*! \} // complex_numbers
 */

/*! \} // numerics
 */
