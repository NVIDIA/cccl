//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  cuda::std::complex<T> z;
  T* a = (T*) &z;
  assert(T(0) == z.real());
  assert(T(0) == z.imag());
  assert(a[0] == z.real());
  assert(a[1] == z.imag());
  a[0] = 5;
  a[1] = 6;
  assert(a[0] == z.real());
  assert(a[1] == z.imag());
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16

  return 0;
}
