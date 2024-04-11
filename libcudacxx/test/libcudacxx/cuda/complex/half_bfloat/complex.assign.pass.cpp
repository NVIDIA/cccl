//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_assignment()
{
  cuda::std::complex<T> from_only_real{static_cast<T>(-1.0), static_cast<T>(1.0)};
  cuda::std::complex<T> from_only_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};
  cuda::std::complex<T> from_real_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};

  const cuda::std::complex<U> only_real{static_cast<U>(42.0), static_cast<U>(0.0)};
  const cuda::std::complex<U> only_imag{static_cast<U>(0.0), static_cast<U>(42.0)};
  const cuda::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(112.0)};

  from_only_real = only_real;
  from_only_imag = only_imag;
  from_real_imag = real_imag;

  assert(from_only_real.real() == static_cast<T>(42.0));
  assert(from_only_real.imag() == static_cast<T>(0.0));
  assert(from_only_imag.real() == static_cast<T>(0.0));
  assert(from_only_imag.imag() == static_cast<T>(42.0));
  assert(from_real_imag.real() == static_cast<T>(42.0));
  assert(from_real_imag.imag() == static_cast<T>(112.0));
}

__host__ __device__ void test()
{
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_assignment<__half, float>();
  test_assignment<__half, double>();
  test_assignment<float, __half>();
  test_assignment<double, __half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_assignment<__nv_bfloat16, float>();
  test_assignment<__nv_bfloat16, double>();
  test_assignment<float, __nv_bfloat16>();
  test_assignment<double, __nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16
}

int main(int arg, char** argv)
{
  test();
  return 0;
}
