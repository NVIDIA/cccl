/*
 *  Copyright 2008-2023 NVIDIA Corporation
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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/complex>
#include <cuda/std/type_traits>

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
  using base = ::cuda::std::complex<T>;
  using base::base;         // inherit constructors
  using base::operator=;    // inherit assignment operators

  complex() = default;

  _CCCL_HOST_DEVICE complex(const base& rhs) : base(rhs) {}
  _CCCL_HOST_DEVICE complex(base&& rhs) : base(::cuda::std::move(rhs)) {}

  template <class U, typename = typename detail::enable_if<!detail::is_same<T, U>::value>::type,
                     typename = typename detail::enable_if<detail::has_promoted_numerical_type<T, U>::value>::type>
  _CCCL_HOST_DEVICE
   complex &operator=(const complex<U> &rhs)
  {
    this->real(static_cast<T>(rhs.real()));
    this->imag(static_cast<T>(rhs.imag()));
    return *this;
  }


  template <class U, typename = typename detail::enable_if<!detail::is_same<T, U>::value>::type,
                     typename = typename detail::enable_if<detail::has_promoted_numerical_type<T, U>::value>::type>
  _CCCL_HOST_DEVICE
  complex &operator=(const U &rhs)
  {
    this->operator=(static_cast<T>(rhs));
    return *this;
  }

  /*! Cast this \p complex to a <tt>std::complex</tt>
   */
  _CCCL_HOST operator ::std::complex<T>() const { return ::std::complex<T>(this->real(), this->imag()); }
};

/* --- Equality Operators --- */

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator==(const complex<T0> &x, const complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p thrust::complex.
 *  \param y The second \p cuda::std::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator==(const complex<T0> &x, const ::cuda::std::complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

/*! Returns true if two \p complex numbers with different underlying type
 * are equal and false otherwise.
 *
 *  \param x The first \p cuda::std::complex.
 *  \param y The second \p thrust::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator==(const ::cuda::std::complex<T0> &x, const complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE bool operator==(const T0 &x, const complex<T1> &y)
{
  return x == y.real() && T0() == y.imag();
}

/*! Returns true if the imaginary part of the \p complex number is zero and
 *  the real part is equal to the scalar. Returns false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE bool operator==(const complex<T0> &x, const T1 &y)
{
  return x.real() == y && x.imag() == T1();
}

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The first \p complex.
 *  \param y The second \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator!=(const complex<T0> &x, const complex<T1> &y)
{
  return !(x == y);
}

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The first \p thrust::complex.
 *  \param y The second \p cuda::std::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator!=(const complex<T0> &x, const ::cuda::std::complex<T1> &y)
{
  return !(x == y);
}

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The second \p cuda::std::complex.
 *  \param y The first \p thrust::complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>>
_CCCL_HOST_DEVICE bool operator!=(const ::cuda::std::complex<T0> &x, const complex<T1> &y)
{
  return !(x == y);
}

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The scalar.
 *  \param y The \p complex.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE bool operator!=(const T0 &x, const complex<T1> &y)
{
  return !(x == y);
}

/*! Returns true if two \p complex numbers with different underlying type
 * are different and false otherwise.
 *
 *  \param x The \p complex.
 *  \param y The scalar.
 *
 *  \tparam \c T0 is convertible to \c T1.
 */
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE bool operator!=(const complex<T0> &x, const T1 &y)
{
  return !(x == y);
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() + y.real(), x.imag() + y.imag());
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() + y, T(x.imag()));
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x + y.real(), T(y.imag()));
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() - y.real(), x.imag() - y.imag());
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() - y, T(x.imag()));
}

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
template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x - y.real(), -T(y.imag()));
}

/* --- Multiplication Operator --- */

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

/* --- Division Operator --- */

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T1>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

template <typename T0,
          typename T1,
          typename = typename detail::enable_if<!detail::is_same<T0, T1>::value>::type,
          typename = typename detail::enable_if<detail::is_arithmetic<T0>::value>::type>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

// The using declarations allows imports all necessary functions for thurst::complex.
// However, they also lead to thrust::abs(1.0F) being valid code after include <thurst/complex.h>.
// We are importing those for the plain value taking overloads and specialize for those taking
// or returning a `thrust::complex` below
using ::cuda::std::abs;
using ::cuda::std::arg;
using ::cuda::std::conj;
using ::cuda::std::norm;
// polar only takes a T but returns a complex<T> so we cannot pull that one in.
// using ::cuda::std::polar;
using ::cuda::std::proj;

using ::cuda::std::exp;
using ::cuda::std::log;
using ::cuda::std::log10;
// pow always returns a complex.
// using ::cuda::std::pow;
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

// Those functions return `cuda::std::complex<T>` so we must provide an explicit overload that returns `thrust::complex<T>`
template<class T>
_CCCL_HOST_DEVICE complex<T> conj(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::conj(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> polar(const T& rho, const T& theta = T{}) {
  return static_cast<complex<T>>(::cuda::std::polar(rho, theta));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> proj(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::proj(c));
}

template<class T>
_CCCL_HOST_DEVICE complex<T> exp(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::exp(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> log(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::log(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> log10(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::log10(c));
}
template<class T0, class T1>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const complex<T1>& y) {
  return static_cast<complex<typename detail::promoted_numerical_type<T0, T1>::type>>(::cuda::std::pow(x, y));
}
template<class T0, class T1, ::cuda::std::__enable_if_t<::cuda::std::is_arithmetic<T1>::value, int> = 0>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const T1& y) {
  return static_cast<complex<typename detail::promoted_numerical_type<T0, T1>::type>>(::cuda::std::pow(x, y));
}
template<class T0, class T1, ::cuda::std::__enable_if_t<::cuda::std::is_arithmetic<T0>::value, int> = 0>
_CCCL_HOST_DEVICE complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const T0& x, const complex<T1>& y) {
  return static_cast<complex<typename detail::promoted_numerical_type<T0, T1>::type>>(::cuda::std::pow(x, y));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> sqrt(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::sqrt(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> acos(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::acos(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> acosh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::acosh(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> asin(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::asin(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> asinh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::asinh(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> atan(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::atan(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> atanh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::atanh(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> cos(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::cos(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> cosh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::cosh(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> sin(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::sin(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> sinh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::sinh(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> tan(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::tan(c));
}
template<class T>
_CCCL_HOST_DEVICE complex<T> tanh(const complex<T>& c) {
  return static_cast<complex<T>>(::cuda::std::tanh(c));
}

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

/*! \} // complex_numbers
 */

/*! \} // numerics
 */
