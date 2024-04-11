//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_void

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_void()
{
  static_assert(cuda::std::is_void<T>::value, "");
  static_assert(cuda::std::is_void<const T>::value, "");
  static_assert(cuda::std::is_void<volatile T>::value, "");
  static_assert(cuda::std::is_void<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_void_v<T>, "");
  static_assert(cuda::std::is_void_v<const T>, "");
  static_assert(cuda::std::is_void_v<volatile T>, "");
  static_assert(cuda::std::is_void_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_void()
{
  static_assert(!cuda::std::is_void<T>::value, "");
  static_assert(!cuda::std::is_void<const T>::value, "");
  static_assert(!cuda::std::is_void<volatile T>::value, "");
  static_assert(!cuda::std::is_void<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_void_v<T>, "");
  static_assert(!cuda::std::is_void_v<const T>, "");
  static_assert(!cuda::std::is_void_v<volatile T>, "");
  static_assert(!cuda::std::is_void_v<const volatile T>, "");
#endif
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
  test_is_void<void>();

  test_is_not_void<int>();
  test_is_not_void<int*>();
  test_is_not_void<int&>();
  test_is_not_void<int&&>();
  test_is_not_void<double>();
  test_is_not_void<const int*>();
  test_is_not_void<char[3]>();
  test_is_not_void<char[]>();
  test_is_not_void<Union>();
  test_is_not_void<Empty>();
  test_is_not_void<bit_zero>();
  test_is_not_void<NotEmpty>();
  test_is_not_void<Abstract>();
  test_is_not_void<Enum>();
  test_is_not_void<FunctionPtr>();
  test_is_not_void<incomplete_type>();

  return 0;
}
