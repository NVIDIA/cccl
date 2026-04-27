// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013, Filipe RNC Maia
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/inverse_trigonometric_functions.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__type_traits/common_type.h>

THRUST_NAMESPACE_BEGIN

/* --- Binary Arithmetic Operators --- */

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator+(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) + static_cast<T>(y.real()),
                    static_cast<T>(x.imag()) + static_cast<T>(y.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator+(const complex<T0>& x, const T1& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) + static_cast<T>(y), static_cast<T>(x.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator+(const T0& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x) + static_cast<T>(y.real()), static_cast<T>(y.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator-(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) - static_cast<T>(y.real()),
                    static_cast<T>(x.imag()) - static_cast<T>(y.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator-(const complex<T0>& x, const T1& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) - static_cast<T>(y), static_cast<T>(x.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator-(const T0& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x) - static_cast<T>(y.real()), -static_cast<T>(y.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator*(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  T xr    = static_cast<T>(x.real());
  T xi    = static_cast<T>(x.imag());
  T yr    = static_cast<T>(y.real());
  T yi    = static_cast<T>(y.imag());
  return complex<T>(xr * yr - xi * yi, xr * yi + xi * yr);
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator*(const complex<T0>& x, const T1& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) * static_cast<T>(y), static_cast<T>(x.imag()) * static_cast<T>(y));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator*(const T0& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x) * static_cast<T>(y.real()), static_cast<T>(x) * static_cast<T>(y.imag()));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator/(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  T s     = ::cuda::std::abs(static_cast<T>(y.real())) + ::cuda::std::abs(static_cast<T>(y.imag()));

  T oos = T(1.0) / s;

  T ars = static_cast<T>(x.real()) * oos;
  T ais = static_cast<T>(x.imag()) * oos;
  T brs = static_cast<T>(y.real()) * oos;
  T bis = static_cast<T>(y.imag()) * oos;

  s = (brs * brs) + (bis * bis);

  oos = T(1.0) / s;

  complex<T> quot(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator/(const complex<T0>& x, const T1& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(static_cast<T>(x.real()) / static_cast<T>(y), static_cast<T>(x.imag()) / static_cast<T>(y));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> operator/(const T0& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(x) / y;
}

/* --- Unary Arithmetic Operators --- */

template <typename T>
_CCCL_HOST_DEVICE complex<T> operator+(const complex<T>& y)
{
  return y;
}

template <typename T>
_CCCL_HOST_DEVICE complex<T> operator-(const complex<T>& y)
{
  return y * -T(1);
}

/* --- Other Basic Arithmetic Functions --- */

// As std::hypot is only C++11 we have to use the C interface
template <typename T>
_CCCL_HOST_DEVICE T abs(const complex<T>& z)
{
  return ::cuda::std::hypot(z.real(), z.imag());
}

template <typename T>
_CCCL_HOST_DEVICE T arg(const complex<T>& z)
{
  return ::cuda::std::atan2(z.imag(), z.real());
}

template <typename T>
_CCCL_HOST_DEVICE complex<T> conj(const complex<T>& z)
{
  return complex<T>(z.real(), -z.imag());
}

template <typename T>
_CCCL_HOST_DEVICE T norm(const complex<T>& z)
{
  return z.real() * z.real() + z.imag() * z.imag();
}

// XXX Why specialize these, we could just rely on ADL.
template <>
_CCCL_HOST_DEVICE inline float norm(const complex<float>& z)
{
  if (::cuda::std::abs(z.real()) < ::cuda::std::sqrt(FLT_MIN)
      && ::cuda::std::abs(z.imag()) < ::cuda::std::sqrt(FLT_MIN))
  {
    float a = z.real() * 4.0f;
    float b = z.imag() * 4.0f;
    return (a * a + b * b) / 16.0f;
  }

  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
_CCCL_HOST_DEVICE inline double norm(const complex<double>& z)
{
  if (::cuda::std::abs(z.real()) < ::cuda::std::sqrt(DBL_MIN)
      && ::cuda::std::abs(z.imag()) < ::cuda::std::sqrt(DBL_MIN))
  {
    double a = z.real() * 4.0;
    double b = z.imag() * 4.0;
    return (a * a + b * b) / 16.0;
  }

  return z.real() * z.real() + z.imag() * z.imag();
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> polar(const T0& m, const T1& theta)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return complex<T>(m * ::cuda::std::cos(theta), m * ::cuda::std::sin(theta));
}

THRUST_NAMESPACE_END
