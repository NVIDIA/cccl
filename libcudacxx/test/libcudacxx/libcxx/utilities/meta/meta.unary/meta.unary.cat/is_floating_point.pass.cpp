//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// keep this test in sync with `is_floating_point.pass.cpp` for `cuda::std::is_floating_point`

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_floating_point()
{
  static_assert(cuda::is_floating_point<T>::value, "");
  static_assert(cuda::is_floating_point<const T>::value, "");
  static_assert(cuda::is_floating_point<volatile T>::value, "");
  static_assert(cuda::is_floating_point<const volatile T>::value, "");
  static_assert(cuda::is_floating_point_v<T>, "");
  static_assert(cuda::is_floating_point_v<const T>, "");
  static_assert(cuda::is_floating_point_v<volatile T>, "");
  static_assert(cuda::is_floating_point_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_floating_point()
{
  static_assert(!cuda::is_floating_point<T>::value, "");
  static_assert(!cuda::is_floating_point<const T>::value, "");
  static_assert(!cuda::is_floating_point<volatile T>::value, "");
  static_assert(!cuda::is_floating_point<const volatile T>::value, "");
  static_assert(!cuda::is_floating_point_v<T>, "");
  static_assert(!cuda::is_floating_point_v<const T>, "");
  static_assert(!cuda::is_floating_point_v<volatile T>, "");
  static_assert(!cuda::is_floating_point_v<const volatile T>, "");
}

class Empty
{};

class NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  __host__ __device__ virtual ~Abstract() = 0;
};

enum Enum
{
  zero,
  one
};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_floating_point<float>();
  test_is_floating_point<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_is_floating_point<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_is_floating_point<__half>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test_is_floating_point<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_floating_point<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_is_floating_point<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_is_floating_point<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_is_floating_point<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_is_floating_point<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_is_floating_point<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  test_is_not_floating_point<short>();
  test_is_not_floating_point<unsigned short>();
  test_is_not_floating_point<int>();
  test_is_not_floating_point<unsigned int>();
  test_is_not_floating_point<long>();
  test_is_not_floating_point<unsigned long>();

  test_is_not_floating_point<cuda::std::nullptr_t>();
  test_is_not_floating_point<void>();
  test_is_not_floating_point<int&>();
  test_is_not_floating_point<int&&>();
  test_is_not_floating_point<int*>();
  test_is_not_floating_point<const int*>();
  test_is_not_floating_point<char[3]>();
  test_is_not_floating_point<char[]>();
  test_is_not_floating_point<Union>();
  test_is_not_floating_point<Empty>();
  test_is_not_floating_point<bit_zero>();
  test_is_not_floating_point<NotEmpty>();
  test_is_not_floating_point<Abstract>();
  test_is_not_floating_point<Enum>();
  test_is_not_floating_point<FunctionPtr>();
  test_is_not_floating_point<incomplete_type>();

  return 0;
}
