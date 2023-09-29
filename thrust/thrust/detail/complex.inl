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

#pragma once

#include <thrust/complex.h>
#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

THRUST_NAMESPACE_BEGIN

/* --- Equality Operators --- */

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const complex<T0> &x, const complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const complex<T0> &x, const ::cuda::std::complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator==(const ::cuda::std::complex<T0> &x, const complex<T1> &y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T0>>>::value,
  bool>::type
operator==(const T0 &x, const complex<T1> &y)
{
  return x == y.real() && T0() == y.imag();
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  bool>::type
operator==(const complex<T0> &x, const T1 &y)
{
  return x.real() == y && x.imag() == T1();
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<detail::is_same<T0, T1>::value, bool>::type
operator!=(const complex<T0> &x, const ::cuda::std::complex<T1> &y)
{
  return !(x == y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::enable_if<detail::is_same<T0, T1>::value, bool>::type
operator!=(const ::cuda::std::complex<T0> &x, const complex<T1> &y)
{
  return !(x == y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T0>>>::value,
  bool>::type
operator!=(const T0 &x, const complex<T1> &y)
{
  return !(x == y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  bool>::type
operator!=(const complex<T0> &x, const T1 &y)
{
  return !(x == y);
}

/* --- Assignment Add Operator --- */

template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator+(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() + y.real(), x.imag() + y.imag());
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator+(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() + y, T(x.imag()));
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator+(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x + y.real(), T(y.imag()));
}

/* --- Substraction Operator --- */

template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator-(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() - y.real(), x.imag() - y.imag());
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator-(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x.real() - y, T(x.imag()));
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator-(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(x - y.real(), -T(y.imag()));
}

/* --- Multiplication Operator --- */

template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator*(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator*(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator*(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of multiplication
  return ::cuda::std::complex<T>(x) * ::cuda::std::complex<T>(y);
}

/* --- Division Operator --- */

template <typename T0, typename T1>
__host__ __device__
  typename detail::disable_if<detail::is_same<T0, T1>::value,
                              complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
  operator/(const complex<T0> &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator/(const complex<T0> &x, const T1 &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

template <typename T0, typename T1>
__host__ __device__ typename detail::disable_if<
  detail::or_<detail::is_same<T0, T1>, detail::not_<detail::is_arithmetic<T1>>>::value,
  complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator/(const T0 &x, const complex<T1> &y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // fall back to std implementation of division
  return ::cuda::std::complex<T>(x) / ::cuda::std::complex<T>(y);
}

THRUST_NAMESPACE_END
