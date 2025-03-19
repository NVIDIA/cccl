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

template <class T>
struct test_data
{
  ::std::complex<T> only_real;
  ::std::complex<T> only_imag;
  ::std::complex<T> real_imag;
};

// we need to disable the execution space check for this test, because std::complex is not available in device code
_CCCL_EXEC_CHECK_DISABLE
template <class T>
__host__ __device__ constexpr auto get_test_data()
{
  test_data<T> data;
  data.only_real = ::std::complex<T>{T(42.0), T()};
  data.only_imag = ::std::complex<T>{T(), T(42.0)};
  data.real_imag = ::std::complex<T>{T(42.0), T(1337.0)};
  return data;
}

template <class T, class U>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_assignment()
{
  constexpr test_data<U> data = get_test_data<U>();

  ::cuda::std::complex<T> only_real{static_cast<T>(-1.0), static_cast<T>(1.0)};
  ::cuda::std::complex<T> only_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};
  ::cuda::std::complex<T> real_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};

  only_real = data.only_real;
  only_imag = data.only_imag;
  real_imag = data.real_imag;

  assert(only_real.real() == static_cast<T>(42.0));
  assert(only_real.imag() == 0);
  assert(only_imag.real() == 0);
  assert(only_imag.imag() == static_cast<T>(42.0));
  assert(real_imag.real() == static_cast<T>(42.0));
  assert(real_imag.imag() == static_cast<T>(1337.0));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_assignment<float, float>();
  test_assignment<float, double>();
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
