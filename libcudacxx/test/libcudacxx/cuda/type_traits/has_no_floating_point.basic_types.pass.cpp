//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__type_traits/has_no_floating_point.h>

#include "cuda_fp_types.h"
#include "test_macros.h"

template <class T>
TEST_FUNC void test_has_no_floating_point()
{
  static_assert(cuda::__has_no_floating_point_v<T>);
  static_assert(cuda::__has_no_floating_point_v<const T>);
  static_assert(cuda::__has_no_floating_point_v<volatile T>);
  static_assert(cuda::__has_no_floating_point_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_has_floating_point()
{
  static_assert(!cuda::__has_no_floating_point_v<T>);
  static_assert(!cuda::__has_no_floating_point_v<const T>);
  static_assert(!cuda::__has_no_floating_point_v<volatile T>);
  static_assert(!cuda::__has_no_floating_point_v<const volatile T>);
}

TEST_FUNC void test_basic_types()
{
  test_has_no_floating_point<bool>();
  test_has_no_floating_point<int>();
  test_has_no_floating_point<unsigned>();
  test_has_no_floating_point<char>();
  test_has_no_floating_point<unsigned char>();
  test_has_no_floating_point<short>();
  test_has_no_floating_point<long long>();

  test_has_floating_point<float>();
  test_has_floating_point<double>();
  NV_IF_TARGET(NV_IS_HOST, (test_has_floating_point<long double>();))

  // arrays
  static_assert(cuda::__has_no_floating_point_v<int[]>);
  static_assert(cuda::__has_no_floating_point_v<const int[]>);
  static_assert(cuda::__has_no_floating_point_v<int[4]>);
  static_assert(cuda::__has_no_floating_point_v<const int[4]>);
  static_assert(!cuda::__has_no_floating_point_v<float[]>);
  static_assert(!cuda::__has_no_floating_point_v<const float[]>);
  static_assert(!cuda::__has_no_floating_point_v<float[4]>);
  static_assert(!cuda::__has_no_floating_point_v<const float[4]>);

#if _CCCL_HAS_CTK()
  test_has_no_floating_point<int4>();
  test_has_no_floating_point<uint2>();
  test_has_floating_point<float2>();
  test_has_floating_point<double2>();
#endif // _CCCL_HAS_CTK()

  // extended floating-point scalar and vector types
#if _CCCL_HAS_NVFP16()
  test_has_floating_point<__half>();
  test_has_floating_point<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_has_floating_point<__nv_bfloat16>();
  test_has_floating_point<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_has_floating_point<__nv_fp8_e4m3>();
  test_has_floating_point<__nv_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
}

int main(int, char**)
{
  test_basic_types();
  return 0;
}
