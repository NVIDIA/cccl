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

template <class T>
_CCCL_GLOBAL_CONSTANT ::std::complex<T> only_real{static_cast<T>(42.0), 0};

template <class T>
_CCCL_GLOBAL_CONSTANT ::std::complex<T> only_imag{0, static_cast<T>(42.0)};

template <class T>
_CCCL_GLOBAL_CONSTANT ::std::complex<T> real_imag{static_cast<T>(42.0), static_cast<T>(1337.0)};

template <class T, class U>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_construction()
{
  const ::cuda::std::complex<T> from_only_real{only_real<U>};
  const ::cuda::std::complex<T> from_only_imag{only_imag<U>};
  const ::cuda::std::complex<T> from_real_imag{real_imag<U>};

  assert(from_only_real.real() == static_cast<T>(42.0));
  assert(from_only_real.imag() == 0);
  assert(from_only_imag.real() == 0);
  assert(from_only_imag.imag() == static_cast<T>(42.0));
  assert(from_real_imag.real() == static_cast<T>(42.0));
  assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_construction<float, float>();
  test_construction<double, float>();
  test_construction<double, double>();

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
