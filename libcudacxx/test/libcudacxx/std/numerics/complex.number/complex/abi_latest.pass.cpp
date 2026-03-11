//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
// class complex
// {
// public:
//   using value_type = T;
//   ...
// };

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  using C = cuda::std::complex<T>;

  static_assert(sizeof(C) == (sizeof(T) * 2), "wrong size");
  static_assert(alignof(C) == (alignof(T) * 2), "misaligned");
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
