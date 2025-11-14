/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
