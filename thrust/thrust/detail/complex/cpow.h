// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013, Filipe RNC Maia
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/cexp.h>
#include <thrust/detail/complex/clog.h>

#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__type_traits/common_type.h>

THRUST_NAMESPACE_BEGIN

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> pow(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return exp(log(complex<T>(x)) * complex<T>(y));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> pow(const complex<T0>& x, const T1& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return exp(log(complex<T>(x)) * T(y));
}

template <typename T0, typename T1>
_CCCL_HOST_DEVICE complex<::cuda::std::common_type_t<T0, T1>> pow(const T0& x, const complex<T1>& y)
{
  using T = ::cuda::std::common_type_t<T0, T1>;
  return exp(::cuda::std::log(T(x)) * complex<T>(y));
}

THRUST_NAMESPACE_END
