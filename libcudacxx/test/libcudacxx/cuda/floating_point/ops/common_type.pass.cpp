//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

#include <cuda/__floating_point_>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T, class U, class Exp>
__host__ __device__ void test_common_type()
{
  ASSERT_SAME_TYPE(cuda::__fp_common_type_t<T, U>, Exp);
  ASSERT_SAME_TYPE(cuda::__fp_common_type_t<U, T>, Exp);
}

int main(int, char**)
{
  // cuda::fp16
  test_common_type<cuda::fp16, cuda::fp16, cuda::fp16>();
  test_common_type<cuda::fp16, cuda::fp32, cuda::fp32>();
  test_common_type<cuda::fp16, cuda::fp64, cuda::fp64>();
  test_common_type<cuda::fp16, float, float>();
  test_common_type<cuda::fp16, double, double>();
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_common_type<cuda::fp16, long double, long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  // cuda::fp32
  test_common_type<cuda::fp32, cuda::fp16, cuda::fp32>();
  test_common_type<cuda::fp32, cuda::fp32, cuda::fp32>();
  test_common_type<cuda::fp32, cuda::fp64, cuda::fp64>();
  test_common_type<cuda::fp32, cuda::bf16, cuda::fp32>();
  test_common_type<cuda::fp32, float, cuda::fp32>();
  test_common_type<cuda::fp32, double, double>();
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_common_type<cuda::fp32, long double, long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  // cuda::fp64
  test_common_type<cuda::fp64, cuda::fp16, cuda::fp64>();
  test_common_type<cuda::fp64, cuda::fp32, cuda::fp64>();
  test_common_type<cuda::fp64, cuda::fp64, cuda::fp64>();
  test_common_type<cuda::fp64, cuda::bf16, cuda::fp64>();
  test_common_type<cuda::fp64, float, cuda::fp64>();
  test_common_type<cuda::fp64, double, cuda::fp64>();
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_common_type<cuda::fp64, long double, long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  // cuda::bf16
  test_common_type<cuda::bf16, cuda::bf16, cuda::bf16>();
  test_common_type<cuda::bf16, cuda::fp32, cuda::fp32>();
  test_common_type<cuda::bf16, cuda::fp64, cuda::fp64>();
  test_common_type<cuda::bf16, float, float>();
  test_common_type<cuda::bf16, double, double>();
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_common_type<cuda::bf16, long double, long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  return 0;
}
