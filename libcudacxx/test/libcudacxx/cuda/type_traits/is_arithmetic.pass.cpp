//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// keep this test in sync with `is_arithmetic.pass.cpp` for `cuda::std::is_arithmetic`

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC void test_is_arithmetic()
{
  static_assert(cuda::is_arithmetic<T>::value);
  static_assert(cuda::is_arithmetic<const T>::value);
  static_assert(cuda::is_arithmetic<volatile T>::value);
  static_assert(cuda::is_arithmetic<const volatile T>::value);
  static_assert(cuda::is_arithmetic_v<T>);
  static_assert(cuda::is_arithmetic_v<const T>);
  static_assert(cuda::is_arithmetic_v<volatile T>);
  static_assert(cuda::is_arithmetic_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_is_not_arithmetic()
{
  static_assert(!cuda::is_arithmetic<T>::value);
  static_assert(!cuda::is_arithmetic<const T>::value);
  static_assert(!cuda::is_arithmetic<volatile T>::value);
  static_assert(!cuda::is_arithmetic<const volatile T>::value);
  static_assert(!cuda::is_arithmetic_v<T>);
  static_assert(!cuda::is_arithmetic_v<const T>);
  static_assert(!cuda::is_arithmetic_v<volatile T>);
  static_assert(!cuda::is_arithmetic_v<const volatile T>);
}

class Empty
{};

class NotEmpty
{
  TEST_FUNC virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  TEST_FUNC virtual ~Abstract() = 0;
};

enum Enum
{
  zero,
  one
};
struct incomplete_type;

using FunctionPtr = void (*)();

int main(int, char**)
{
  test_is_arithmetic<float>();
  test_is_arithmetic<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_is_arithmetic<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_is_arithmetic<__half>();
  test_is_not_arithmetic<__half2>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test_is_arithmetic<__nv_bfloat16>();
  test_is_not_arithmetic<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_arithmetic<__nv_fp8_e4m3>();
  test_is_not_arithmetic<__nv_fp8x2_e4m3>();
  test_is_not_arithmetic<__nv_fp8x4_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_is_arithmetic<__nv_fp8_e5m2>();
  test_is_not_arithmetic<__nv_fp8x2_e5m2>();
  test_is_not_arithmetic<__nv_fp8x4_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_is_arithmetic<__nv_fp8_e8m0>();
  test_is_not_arithmetic<__nv_fp8x2_e8m0>();
  test_is_not_arithmetic<__nv_fp8x4_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_is_arithmetic<__nv_fp6_e2m3>();
  test_is_not_arithmetic<__nv_fp6x2_e2m3>();
  test_is_not_arithmetic<__nv_fp6x4_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_is_arithmetic<__nv_fp6_e3m2>();
  test_is_not_arithmetic<__nv_fp6x2_e3m2>();
  test_is_not_arithmetic<__nv_fp6x4_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_is_arithmetic<__nv_fp4_e2m1>();
  test_is_not_arithmetic<__nv_fp4x2_e2m1>();
  test_is_not_arithmetic<__nv_fp4x4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  test_is_arithmetic<short>();
  test_is_arithmetic<unsigned short>();
  test_is_arithmetic<int>();
  test_is_arithmetic<unsigned int>();
  test_is_arithmetic<long>();
  test_is_arithmetic<unsigned long>();
  test_is_arithmetic<long long>();
  test_is_arithmetic<unsigned long long>();
  test_is_arithmetic<bool>();
  test_is_arithmetic<char>();
  test_is_arithmetic<signed char>();
  test_is_arithmetic<unsigned char>();
  test_is_arithmetic<wchar_t>();
#if _CCCL_HAS_CHAR8_T()
  test_is_arithmetic<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_is_arithmetic<char16_t>();
  test_is_arithmetic<char32_t>();
#if _CCCL_HAS_INT128()
  test_is_arithmetic<__int128_t>();
  test_is_arithmetic<__uint128_t>();
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_FLOAT128()
  test_is_arithmetic<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_is_not_arithmetic<cuda::std::nullptr_t>();
  test_is_not_arithmetic<void>();
  test_is_not_arithmetic<float&>();
  test_is_not_arithmetic<float&&>();
  test_is_not_arithmetic<float*>();
  test_is_not_arithmetic<void()>();
  test_is_not_arithmetic<int Empty::*>();
  test_is_not_arithmetic<int&>();
  test_is_not_arithmetic<int&&>();
  test_is_not_arithmetic<int*>();
  test_is_not_arithmetic<const int*>();
  test_is_not_arithmetic<char[3]>();
  test_is_not_arithmetic<char[]>();
  test_is_not_arithmetic<Union>();
  test_is_not_arithmetic<Empty>();
  test_is_not_arithmetic<bit_zero>();
  test_is_not_arithmetic<NotEmpty>();
  test_is_not_arithmetic<Abstract>();
  test_is_not_arithmetic<Enum>();
  test_is_not_arithmetic<FunctionPtr>();
  test_is_not_arithmetic<incomplete_type>();

  return 0;
}
