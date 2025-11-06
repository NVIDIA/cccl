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
#include <thrust/detail/complex/math_private.h>

#include <cuda/std/cmath>
#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace detail::complex
{
_CCCL_HOST_DEVICE inline complex<float> cprojf(const complex<float>& z)
{
  if (!isinf(z.real()) && !isinf(z.imag()))
  {
    return z;
  }
  else
  {
    // ::cuda::std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<float>(::cuda::std::numeric_limits<float>::infinity(), copysignf(0.0, z.imag()));
  }
}

_CCCL_HOST_DEVICE inline complex<double> cproj(const complex<double>& z)
{
  if (!isinf(z.real()) && !isinf(z.imag()))
  {
    return z;
  }
  else
  {
    // ::cuda::std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<double>(::cuda::std::numeric_limits<double>::infinity(), copysign(0.0, z.imag()));
  }
}
} // namespace detail::complex

template <typename T>
_CCCL_HOST_DEVICE inline thrust::complex<T> proj(const thrust::complex<T>& z)
{
  return detail::complex::cproj(z);
}

template <>
_CCCL_HOST_DEVICE inline thrust::complex<double> proj(const thrust::complex<double>& z)
{
  return detail::complex::cproj(z);
}

template <>
_CCCL_HOST_DEVICE inline thrust::complex<float> proj(const thrust::complex<float>& z)
{
  return detail::complex::cprojf(z);
}

THRUST_NAMESPACE_END
