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
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_comparison()
{
  const ::cuda::std::complex<T> input{static_cast<T>(-1.0), static_cast<T>(1.0)};

  constexpr ::std::complex<U> not_equal_real{static_cast<U>(-1.0), U{}};
  constexpr ::std::complex<U> not_equal_imag{U{}, static_cast<U>(1.0)};
  constexpr ::std::complex<U> equal{static_cast<U>(-1.0), static_cast<U>(1.0)};

  assert(!(input == not_equal_real));
  assert(!(input == not_equal_imag));
  assert(input == equal);

  assert(input != not_equal_real);
  assert(input != not_equal_imag);
  assert(!(input != equal));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_comparison<float, float>();
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
