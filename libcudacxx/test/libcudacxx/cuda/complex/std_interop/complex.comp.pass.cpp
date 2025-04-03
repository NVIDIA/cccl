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
  ::std::complex<T> not_equal_real;
  ::std::complex<T> not_equal_imag;
  ::std::complex<T> equal;
};

// we need to disable the execution space check for this test, because std::complex is not available in device code
_CCCL_EXEC_CHECK_DISABLE
template <class T>
__host__ __device__ constexpr auto get_test_data()
{
  test_data<T> data;
  data.not_equal_real = ::std::complex<T>{T(-1.0), T()};
  data.not_equal_imag = ::std::complex<T>{T(), T(1.0)};
  data.equal          = ::std::complex<T>{T(-1.0), T(1.0)};
  return data;
}

template <class T, class U>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_comparison()
{
  constexpr test_data<U> data = get_test_data<U>();

  const ::cuda::std::complex<T> input{-1.f, 1.f};

  assert(!(input == data.not_equal_real));
  assert(!(input == data.not_equal_imag));
  assert(input == data.equal);

  assert(!(data.not_equal_real == input));
  assert(!(data.not_equal_imag == input));
  assert(data.equal == input);

  assert(input != data.not_equal_real);
  assert(input != data.not_equal_imag);
  assert(!(input != data.equal));

  assert(data.not_equal_real != input);
  assert(data.not_equal_imag != input);
  assert(!(data.equal != input));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_comparison<float, float>();
  test_comparison<float, double>();
  test_comparison<double, float>();
  test_comparison<double, double>();

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
