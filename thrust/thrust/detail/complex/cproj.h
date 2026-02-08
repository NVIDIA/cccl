// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013, Filipe RNC Maia
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace detail::complex
{
_CCCL_HOST_DEVICE inline complex<float> cprojf(const complex<float>& z)
{
  if (!::cuda::std::isinf(z.real()) && !::cuda::std::isinf(z.imag()))
  {
    return z;
  }
  else
  {
    return complex<float>(::cuda::std::numeric_limits<float>::infinity(), ::cuda::std::copysignf(0.0, z.imag()));
  }
}

_CCCL_HOST_DEVICE inline complex<double> cproj(const complex<double>& z)
{
  if (!::cuda::std::isinf(z.real()) && !::cuda::std::isinf(z.imag()))
  {
    return z;
  }
  else
  {
    return complex<double>(::cuda::std::numeric_limits<double>::infinity(), ::cuda::std::copysign(0.0, z.imag()));
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
