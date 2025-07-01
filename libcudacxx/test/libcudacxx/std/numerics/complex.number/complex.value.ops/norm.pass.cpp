//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   T
//   norm(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test()
{
  cuda::std::complex<T> z(3, 4);
  assert(norm(z) == T(25));

  return true;
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    T r = norm(testcases[i]);
    switch (classify(testcases[i]))
    {
      case zero:
        assert(r == T(0));
        assert(!cuda::std::signbit(r));
        break;
      case non_zero:
        printf("%f %f\n", float(testcases[i].real()), float(testcases[i].imag()));
        assert(cuda::std::isfinite(r) && r > T(0));
        break;
      case inf:
        assert(cuda::std::isinf(r) && r > T(0));
        break;
      case NaN:
        assert(cuda::std::isnan(r));
        break;
      case non_zero_nan:
        assert(cuda::std::isnan(r));
        break;
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_CONSTEXPR_CMATH_TRAITS()
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
#  if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>(), "");
#  endif // _CCCL_HAS_LONG_DOUBLE()
#endif // _CCCL_HAS_CONSTEXPR_CMATH_TRAITS()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
