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
__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS void test_conversion()
{
  const ::cuda::std::complex<T> from_only_real{static_cast<T>(42.0), 0};
  const ::cuda::std::complex<T> from_only_imag{0, static_cast<T>(42.0)};
  const ::cuda::std::complex<T> from_real_imag{static_cast<T>(42.0), static_cast<T>(1337.0)};

  // Construct std::complex from cuda::std::complex
  const ::std::complex<T> only_real{from_only_real};
  const ::std::complex<T> only_imag{from_only_imag};
  const ::std::complex<T> real_imag{from_real_imag};

  // To check that the conversion is correct, we convert back to cuda::std::complex
  const ::cuda::std::complex<T> only_real_copy{only_real};
  const ::cuda::std::complex<T> only_imag_copy{only_imag};
  const ::cuda::std::complex<T> real_imag_copy{real_imag};

  assert(only_real_copy.real() == static_cast<T>(42.0));
  assert(only_real_copy.imag() == 0);
  assert(only_imag_copy.real() == 0);
  assert(only_imag_copy.imag() == static_cast<T>(42.0));
  assert(real_imag_copy.real() == static_cast<T>(42.0));
  assert(real_imag_copy.imag() == static_cast<T>(1337.0));
}

__host__ __device__ _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS bool test()
{
  test_conversion<float>();
  test_conversion<double>();

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
