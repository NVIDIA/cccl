//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include <complex>

#include "test_macros.h"
#include <nv/target>

_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_assignment()
{
  ::cuda::std::complex<T> from_only_real{static_cast<T>(-1.0), static_cast<T>(1.0)};
  ::cuda::std::complex<T> from_only_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};
  ::cuda::std::complex<T> from_real_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};

  constexpr ::std::complex<U> only_real{static_cast<U>(42.0), U{}};
  constexpr ::std::complex<U> only_imag{U{}, static_cast<U>(42.0)};
  constexpr ::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(1337.0)};

  from_only_real = only_real;
  from_only_imag = only_imag;
  from_real_imag = real_imag;

  assert(from_only_real.real() == static_cast<T>(42.0));
  assert(from_only_real.imag() == 0);
  assert(from_only_imag.real() == 0);
  assert(from_only_imag.imag() == static_cast<T>(42.0));
  assert(from_real_imag.real() == static_cast<T>(42.0));
  assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_assignment<float, float>();
  test_assignment<double, float>();
  test_assignment<double, double>();

  return true;
}

int main(int arg, char** argv)
{
  test();
#if _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS()
  static_assert(test());
#endif // _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS()

  return 0;
}
