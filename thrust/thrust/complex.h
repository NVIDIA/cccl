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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/complex>

#include <complex>
#include <iostream>

THRUST_NAMESPACE_BEGIN

using ::cuda::std::complex;

template <typename T>
struct proclaim_trivially_relocatable<complex<T>> : true_type
{};

using ::cuda::std::abs;
using ::cuda::std::arg;
using ::cuda::std::conj;
using ::cuda::std::norm;
using ::cuda::std::polar;

using ::cuda::std::cos;
using ::cuda::std::cosh;
using ::cuda::std::sin;
using ::cuda::std::sinh;
using ::cuda::std::tan;
using ::cuda::std::tanh;

using ::cuda::std::acos;
using ::cuda::std::acosh;
using ::cuda::std::asin;
using ::cuda::std::asinh;
using ::cuda::std::atan;
using ::cuda::std::atanh;

using ::cuda::std::exp;
using ::cuda::std::log;
using ::cuda::std::log10;

using ::cuda::std::pow;
using ::cuda::std::sqrt;

using ::cuda::std::proj;

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator==(const complex<T0>& x, const complex<T1>& y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator==(const T0& x, const complex<T1>& y)
{
  return x == y.real() && y.imag() == T1();
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator==(const complex<T0>& x, const T1& y)
{
  return x.real() == y && x.imag() == T1();
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator!=(const complex<T0>& x, const complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator!=(const T0& x, const complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE bool operator!=(const complex<T0>& x, const T1& y)
{
  return !(x == y);
}

_LIBCUDACXX_END_NAMESPACE_STD
