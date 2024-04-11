//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_arithmetic

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_arithmetic()
{
  static_assert(cuda::std::is_arithmetic<T>::value, "");
  static_assert(cuda::std::is_arithmetic<const T>::value, "");
  static_assert(cuda::std::is_arithmetic<volatile T>::value, "");
  static_assert(cuda::std::is_arithmetic<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_arithmetic_v<T>, "");
  static_assert(cuda::std::is_arithmetic_v<const T>, "");
  static_assert(cuda::std::is_arithmetic_v<volatile T>, "");
  static_assert(cuda::std::is_arithmetic_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_arithmetic()
{
  static_assert(!cuda::std::is_arithmetic<T>::value, "");
  static_assert(!cuda::std::is_arithmetic<const T>::value, "");
  static_assert(!cuda::std::is_arithmetic<volatile T>::value, "");
  static_assert(!cuda::std::is_arithmetic<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_arithmetic_v<T>, "");
  static_assert(!cuda::std::is_arithmetic_v<const T>, "");
  static_assert(!cuda::std::is_arithmetic_v<volatile T>, "");
  static_assert(!cuda::std::is_arithmetic_v<const volatile T>, "");
#endif
}

class incomplete_type;

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

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_arithmetic<short>();
  test_is_arithmetic<unsigned short>();
  test_is_arithmetic<int>();
  test_is_arithmetic<unsigned int>();
  test_is_arithmetic<long>();
  test_is_arithmetic<unsigned long>();
  test_is_arithmetic<bool>();
  test_is_arithmetic<char>();
  test_is_arithmetic<signed char>();
  test_is_arithmetic<unsigned char>();
  test_is_arithmetic<wchar_t>();
  test_is_arithmetic<double>();

  test_is_not_arithmetic<cuda::std::nullptr_t>();
  test_is_not_arithmetic<void>();
  test_is_not_arithmetic<int&>();
  test_is_not_arithmetic<int&&>();
  test_is_not_arithmetic<int*>();
  test_is_not_arithmetic<const int*>();
  test_is_not_arithmetic<char[3]>();
  test_is_not_arithmetic<char[]>();
  test_is_not_arithmetic<Union>();
  test_is_not_arithmetic<Enum>();
  test_is_not_arithmetic<FunctionPtr>();
  test_is_not_arithmetic<Empty>();
  test_is_not_arithmetic<incomplete_type>();
  test_is_not_arithmetic<bit_zero>();
  test_is_not_arithmetic<NotEmpty>();
  test_is_not_arithmetic<Abstract>();

  return 0;
}
