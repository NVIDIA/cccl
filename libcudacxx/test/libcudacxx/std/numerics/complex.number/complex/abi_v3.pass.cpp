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
//   typedef T value_type;
//   ...
// };

#define _LIBCUDACXX_CUDA_ABI_VERSION 3

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  typedef cuda::std::complex<T> C;

  static_assert(sizeof(C) == (sizeof(T) * 2), "wrong size");
  static_assert(alignof(C) == (alignof(T)), "misaligned");
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
// These specializations didn't exist at the time of ABI v3:
#ifdef _LIBCUDACXX_HAS_NVFP16
//    test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
//    test<__nv_bfloat16>();
#endif

  return 0;
}
