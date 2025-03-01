//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_assignment(cuda::std::complex<U> v = {})
{
  cuda::std::complex<T> converting(v);

  cuda::std::complex<T> assigning{};
  assigning = v;
}

__host__ __device__ void test()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_assignment<__half, float>();
  test_assignment<__half, double>();
  test_assignment<float, __half>();
  test_assignment<double, __half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_assignment<__nv_bfloat16, float>();
  test_assignment<__nv_bfloat16, double>();
  test_assignment<float, __nv_bfloat16>();
  test_assignment<double, __nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int arg, char** argv)
{
  test();
  return 0;
}
