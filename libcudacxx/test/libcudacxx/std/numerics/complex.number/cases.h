//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <cuda/std/cassert>
#include <cuda/std/complex>

template <class T>
using testcases_t = cuda::std::complex<T>[152];

template <class T>
struct _testcases
{
  testcases_t<T> _cases;

  static constexpr size_t count = sizeof(testcases_t<T>) / sizeof(cuda::std::complex<T>);

  __host__ __device__ const cuda::std::complex<T>* begin() const
  {
    return &_cases[0];
  }
  __host__ __device__ const cuda::std::complex<T>* cbegin() const
  {
    return &_cases[0];
  }
  __host__ __device__ cuda::std::complex<T>* begin()
  {
    return &_cases[0];
  }

  __host__ __device__ const cuda::std::complex<T>* end() const
  {
    return &_cases[count];
  }
  __host__ __device__ const cuda::std::complex<T>* cend() const
  {
    return &_cases[count];
  }
  __host__ __device__ cuda::std::complex<T>* end()
  {
    return &_cases[count];
  }

  __host__ __device__ cuda::std::complex<T>& operator[](size_t n)
  {
    return _cases[n];
  }

  __host__ __device__ const cuda::std::complex<T>& operator[](size_t n) const
  {
    return _cases[n];
  }
};

template <class T>
__host__ __device__ _testcases<T> get_testcases()
{
  _testcases<T> tc{
    cuda::std::complex<T>(1.e-2, 1.e-2),
    cuda::std::complex<T>(-1.e-2, 1.e-2),
    cuda::std::complex<T>(-1.e-2, -1.e-2),
    cuda::std::complex<T>(1.e-2, -1.e-2),

    cuda::std::complex<T>(1.e+2, 1.e-2),
    cuda::std::complex<T>(-1.e+2, 1.e-2),
    cuda::std::complex<T>(-1.e+2, -1.e-2),
    cuda::std::complex<T>(1.e+2, -1.e-2),

    cuda::std::complex<T>(1.e-2, 1.e+2),
    cuda::std::complex<T>(-1.e-2, 1.e+2),
    cuda::std::complex<T>(-1.e-2, -1.e+2),
    cuda::std::complex<T>(1.e-2, -1.e+2),

    cuda::std::complex<T>(1.e+2, 1.e+2),
    cuda::std::complex<T>(-1.e+2, 1.e+2),
    cuda::std::complex<T>(-1.e+2, -1.e+2),
    cuda::std::complex<T>(1.e+2, -1.e+2),

    cuda::std::complex<T>(-0, -1.e-2),
    cuda::std::complex<T>(-0, 1.e-2),
    cuda::std::complex<T>(-0, 1.e+2),
    cuda::std::complex<T>(-0, -1.e+2),
    cuda::std::complex<T>(0, -1.e-2),
    cuda::std::complex<T>(0, 1.e-2),
    cuda::std::complex<T>(0, 1.e+2),
    cuda::std::complex<T>(0, -1.e+2),

    cuda::std::complex<T>(-1.e-2, -0),
    cuda::std::complex<T>(1.e-2, -0),
    cuda::std::complex<T>(1.e+2, -0),
    cuda::std::complex<T>(-1.e+2, -0),
    cuda::std::complex<T>(-1.e-2, 0),
    cuda::std::complex<T>(1.e-2, 0),
    cuda::std::complex<T>(1.e+2, 0),

    cuda::std::complex<T>(NAN, NAN),
    cuda::std::complex<T>(-INFINITY, NAN),
    cuda::std::complex<T>(-2, NAN),
    cuda::std::complex<T>(-1, NAN),
    cuda::std::complex<T>(-0.5, NAN),
    cuda::std::complex<T>(-0., NAN),
    cuda::std::complex<T>(+0., NAN),
    cuda::std::complex<T>(0.5, NAN),
    cuda::std::complex<T>(1, NAN),
    cuda::std::complex<T>(2, NAN),
    cuda::std::complex<T>(INFINITY, NAN),

    cuda::std::complex<T>(NAN, -INFINITY),
    cuda::std::complex<T>(-INFINITY, -INFINITY),
    cuda::std::complex<T>(-2, -INFINITY),
    cuda::std::complex<T>(-1, -INFINITY),
    cuda::std::complex<T>(-0.5, -INFINITY),
    cuda::std::complex<T>(-0., -INFINITY),
    cuda::std::complex<T>(+0., -INFINITY),
    cuda::std::complex<T>(0.5, -INFINITY),
    cuda::std::complex<T>(1, -INFINITY),
    cuda::std::complex<T>(2, -INFINITY),
    cuda::std::complex<T>(INFINITY, -INFINITY),

    cuda::std::complex<T>(NAN, -2),
    cuda::std::complex<T>(-INFINITY, -2),
    cuda::std::complex<T>(-2, -2),
    cuda::std::complex<T>(-1, -2),
    cuda::std::complex<T>(-0.5, -2),
    cuda::std::complex<T>(-0., -2),
    cuda::std::complex<T>(+0., -2),
    cuda::std::complex<T>(0.5, -2),
    cuda::std::complex<T>(1, -2),
    cuda::std::complex<T>(2, -2),
    cuda::std::complex<T>(INFINITY, -2),

    cuda::std::complex<T>(NAN, -1),
    cuda::std::complex<T>(-INFINITY, -1),
    cuda::std::complex<T>(-2, -1),
    cuda::std::complex<T>(-1, -1),
    cuda::std::complex<T>(-0.5, -1),
    cuda::std::complex<T>(-0., -1),
    cuda::std::complex<T>(+0., -1),
    cuda::std::complex<T>(0.5, -1),
    cuda::std::complex<T>(1, -1),
    cuda::std::complex<T>(2, -1),
    cuda::std::complex<T>(INFINITY, -1),

    cuda::std::complex<T>(NAN, -0.5),
    cuda::std::complex<T>(-INFINITY, -0.5),
    cuda::std::complex<T>(-2, -0.5),
    cuda::std::complex<T>(-1, -0.5),
    cuda::std::complex<T>(-0.5, -0.5),
    cuda::std::complex<T>(-0., -0.5),
    cuda::std::complex<T>(+0., -0.5),
    cuda::std::complex<T>(0.5, -0.5),
    cuda::std::complex<T>(1, -0.5),
    cuda::std::complex<T>(2, -0.5),
    cuda::std::complex<T>(INFINITY, -0.5),

    cuda::std::complex<T>(NAN, -0.),
    cuda::std::complex<T>(-INFINITY, -0.),
    cuda::std::complex<T>(-2, -0.),
    cuda::std::complex<T>(-1, -0.),
    cuda::std::complex<T>(-0.5, -0.),
    cuda::std::complex<T>(-0., -0.),
    cuda::std::complex<T>(+0., -0.),
    cuda::std::complex<T>(0.5, -0.),
    cuda::std::complex<T>(1, -0.),
    cuda::std::complex<T>(2, -0.),
    cuda::std::complex<T>(INFINITY, -0.),

    cuda::std::complex<T>(NAN, +0.),
    cuda::std::complex<T>(-INFINITY, +0.),
    cuda::std::complex<T>(-2, +0.),
    cuda::std::complex<T>(-1, +0.),
    cuda::std::complex<T>(-0.5, +0.),
    cuda::std::complex<T>(-0., +0.),
    cuda::std::complex<T>(+0., +0.),
    cuda::std::complex<T>(0.5, +0.),
    cuda::std::complex<T>(1, +0.),
    cuda::std::complex<T>(2, +0.),
    cuda::std::complex<T>(INFINITY, +0.),

    cuda::std::complex<T>(NAN, 0.5),
    cuda::std::complex<T>(-INFINITY, 0.5),
    cuda::std::complex<T>(-2, 0.5),
    cuda::std::complex<T>(-1, 0.5),
    cuda::std::complex<T>(-0.5, 0.5),
    cuda::std::complex<T>(-0., 0.5),
    cuda::std::complex<T>(+0., 0.5),
    cuda::std::complex<T>(0.5, 0.5),
    cuda::std::complex<T>(1, 0.5),
    cuda::std::complex<T>(2, 0.5),
    cuda::std::complex<T>(INFINITY, 0.5),

    cuda::std::complex<T>(NAN, 1),
    cuda::std::complex<T>(-INFINITY, 1),
    cuda::std::complex<T>(-2, 1),
    cuda::std::complex<T>(-1, 1),
    cuda::std::complex<T>(-0.5, 1),
    cuda::std::complex<T>(-0., 1),
    cuda::std::complex<T>(+0., 1),
    cuda::std::complex<T>(0.5, 1),
    cuda::std::complex<T>(1, 1),
    cuda::std::complex<T>(2, 1),
    cuda::std::complex<T>(INFINITY, 1),

    cuda::std::complex<T>(NAN, 2),
    cuda::std::complex<T>(-INFINITY, 2),
    cuda::std::complex<T>(-2, 2),
    cuda::std::complex<T>(-1, 2),
    cuda::std::complex<T>(-0.5, 2),
    cuda::std::complex<T>(-0., 2),
    cuda::std::complex<T>(+0., 2),
    cuda::std::complex<T>(0.5, 2),
    cuda::std::complex<T>(1, 2),
    cuda::std::complex<T>(2, 2),
    cuda::std::complex<T>(INFINITY, 2),

    cuda::std::complex<T>(NAN, INFINITY),
    cuda::std::complex<T>(-INFINITY, INFINITY),
    cuda::std::complex<T>(-2, INFINITY),
    cuda::std::complex<T>(-1, INFINITY),
    cuda::std::complex<T>(-0.5, INFINITY),
    cuda::std::complex<T>(-0., INFINITY),
    cuda::std::complex<T>(+0., INFINITY),
    cuda::std::complex<T>(0.5, INFINITY),
    cuda::std::complex<T>(1, INFINITY),
    cuda::std::complex<T>(2, INFINITY),
    cuda::std::complex<T>(INFINITY, INFINITY)};

  return tc;
}

enum
{
  zero,
  non_zero,
  inf,
  NaN,
  non_zero_nan
};

template <class T>
__host__ __device__ int classify(const cuda::std::complex<T>& x)
{
  if (x == cuda::std::complex<T>())
  {
    return zero;
  }
  if (cuda::std::isinf(x.real()) || cuda::std::isinf(x.imag()))
  {
    return inf;
  }
  if (cuda::std::isnan(x.real()) && cuda::std::isnan(x.imag()))
  {
    return NaN;
  }
  if (cuda::std::isnan(x.real()))
  {
    if (x.imag() == T(0))
    {
      return NaN;
    }
    return non_zero_nan;
  }
  if (cuda::std::isnan(x.imag()))
  {
    if (x.real() == T(0))
    {
      return NaN;
    }
    return non_zero_nan;
  }
  return non_zero;
}

template <class T>
inline __host__ __device__ int classify(T x)
{
  if (x == T(0))
  {
    return zero;
  }
  if (cuda::std::isinf(x))
  {
    return inf;
  }
  if (cuda::std::isnan(x))
  {
    return NaN;
  }
  return non_zero;
}

__host__ __device__ void is_about(float x, float y)
{
  assert(cuda::std::abs((x - y) / (x + y)) < 1.e-6);
}

__host__ __device__ void is_about(double x, double y)
{
  assert(cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}

// CUDA treats long double as double
/*
__host__ __device__ void is_about(long double x, long double y)
{
    assert(cuda::std::abs((x-y)/(x+y)) < 1.e-14);
}
*/

#ifdef _LIBCUDACXX_HAS_NVFP16
__host__ __device__ void is_about(__half x, __half y)
{
  assert(cuda::std::fabs((x - y) / (x + y)) <= __half(1e-3));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#ifdef _LIBCUDACXX_HAS_NVBF16
__host__ __device__ void is_about(__nv_bfloat16 x, __nv_bfloat16 y)
{
  assert(cuda::std::fabs((x - y) / (x + y)) <= __nv_bfloat16(5e-3));
}
#endif // _LIBCUDACXX_HAS_NVBF16

#endif // CASES_H
