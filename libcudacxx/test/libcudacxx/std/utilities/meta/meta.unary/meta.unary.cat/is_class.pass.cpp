//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_class

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_class()
{
  static_assert(cuda::std::is_class<T>::value, "");
  static_assert(cuda::std::is_class<const T>::value, "");
  static_assert(cuda::std::is_class<volatile T>::value, "");
  static_assert(cuda::std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_class_v<T>, "");
  static_assert(cuda::std::is_class_v<const T>, "");
  static_assert(cuda::std::is_class_v<volatile T>, "");
  static_assert(cuda::std::is_class_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_class()
{
  static_assert(!cuda::std::is_class<T>::value, "");
  static_assert(!cuda::std::is_class<const T>::value, "");
  static_assert(!cuda::std::is_class<volatile T>::value, "");
  static_assert(!cuda::std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_class_v<T>, "");
  static_assert(!cuda::std::is_class_v<const T>, "");
  static_assert(!cuda::std::is_class_v<volatile T>, "");
  static_assert(!cuda::std::is_class_v<const volatile T>, "");
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
  test_is_class<Empty>();
  test_is_class<bit_zero>();
  test_is_class<NotEmpty>();
  test_is_class<Abstract>();
  test_is_class<incomplete_type>();

  // In C++03 we have an emulation of cuda::std::nullptr_t
  test_is_not_class<cuda::std::nullptr_t>();
  test_is_not_class<void>();
  test_is_not_class<int>();
  test_is_not_class<int&>();
  test_is_not_class<int&&>();
  test_is_not_class<int*>();
  test_is_not_class<double>();
  test_is_not_class<const int*>();
  test_is_not_class<char[3]>();
  test_is_not_class<char[]>();
  test_is_not_class<Enum>();
  test_is_not_class<Union>();
  test_is_not_class<FunctionPtr>();

  return 0;
}
