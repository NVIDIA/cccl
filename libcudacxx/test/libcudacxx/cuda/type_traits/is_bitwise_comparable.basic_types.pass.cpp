//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_bitwise_comparable()
{
  static_assert(cuda::is_bitwise_comparable<T>::value);
  static_assert(cuda::is_bitwise_comparable_v<T>);
  static_assert(cuda::is_bitwise_comparable_v<const T>);
  static_assert(cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(cuda::is_bitwise_comparable_v<const volatile T>);
}

template <class T>
__host__ __device__ void test_is_not_bitwise_comparable()
{
  static_assert(!cuda::is_bitwise_comparable<T>::value);
  static_assert(!cuda::is_bitwise_comparable_v<T>);
  static_assert(!cuda::is_bitwise_comparable_v<const T>);
  static_assert(!cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(!cuda::is_bitwise_comparable_v<const volatile T>);
}

__host__ __device__ void test_basic_types()
{
  // types with unique object representations
  test_is_bitwise_comparable<bool>();
  test_is_bitwise_comparable<int>();
  test_is_bitwise_comparable<unsigned>();
  test_is_bitwise_comparable<char>();
  test_is_bitwise_comparable<unsigned char>();
  test_is_bitwise_comparable<short>();
  test_is_bitwise_comparable<long long>();

  // types without unique object representations
  test_is_not_bitwise_comparable<float>();
  test_is_not_bitwise_comparable<double>();

  // arrays
  static_assert(cuda::is_bitwise_comparable_v<int[]>);
  static_assert(cuda::is_bitwise_comparable_v<const int[]>);
  static_assert(cuda::is_bitwise_comparable_v<int[4]>);
  static_assert(cuda::is_bitwise_comparable_v<const int[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<float[]>);
  static_assert(!cuda::is_bitwise_comparable_v<const float[]>);

  // extended floating-point scalar types
#if _CCCL_HAS_NVFP16()
  test_is_not_bitwise_comparable<__half>();
  test_is_not_bitwise_comparable<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_not_bitwise_comparable<__nv_bfloat16>();
  test_is_not_bitwise_comparable<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_not_bitwise_comparable<__nv_fp8_e4m3>();
  test_is_not_bitwise_comparable<__nv_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
}

int main(int, char**)
{
  test_basic_types();
  return 0;
}
