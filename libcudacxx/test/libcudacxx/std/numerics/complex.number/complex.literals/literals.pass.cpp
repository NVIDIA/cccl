//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// UNSUPPORTED: gcc-7

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using namespace cuda::std::literals::complex_literals;

  static_assert(cuda::std::is_same_v<decltype(3.0if), cuda::std::complex<float>>);
  static_assert(cuda::std::is_same_v<decltype(3if), cuda::std::complex<float>>);
  static_assert(cuda::std::is_same_v<decltype(3.0i), cuda::std::complex<double>>);
  static_assert(cuda::std::is_same_v<decltype(3i), cuda::std::complex<double>>);
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::is_same_v<decltype(3.0il), cuda::std::complex<long double>>);
  static_assert(cuda::std::is_same_v<decltype(3il), cuda::std::complex<long double>>);
#endif // _CCCL_HAS_LONG_DOUBLE()

  {
    cuda::std::complex<float> c1 = 3.0if;
    assert((c1 == cuda::std::complex<float>{0.0f, 3.0f}));
    auto c2 = 3if;
    assert(c1 == c2);
  }

  {
    cuda::std::complex<double> c1 = 3.0i;
    assert((c1 == cuda::std::complex<double>{0.0f, 3.0}));
    auto c2 = 3i;
    assert(c1 == c2);
  }

#if _CCCL_HAS_LONG_DOUBLE()
  {
    cuda::std::complex<long double> c1 = 3.0il;
    assert((c1 == cuda::std::complex<long double>{0.0f, 3.0l}));
    auto c2 = 3il;
    assert(c1 == c2);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
