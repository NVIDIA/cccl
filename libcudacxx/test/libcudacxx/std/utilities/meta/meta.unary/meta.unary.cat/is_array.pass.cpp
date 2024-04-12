//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_array

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_array()
{
  static_assert(cuda::std::is_array<T>::value, "");
  static_assert(cuda::std::is_array<const T>::value, "");
  static_assert(cuda::std::is_array<volatile T>::value, "");
  static_assert(cuda::std::is_array<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_array_v<T>, "");
  static_assert(cuda::std::is_array_v<const T>, "");
  static_assert(cuda::std::is_array_v<volatile T>, "");
  static_assert(cuda::std::is_array_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_array()
{
  static_assert(!cuda::std::is_array<T>::value, "");
  static_assert(!cuda::std::is_array<const T>::value, "");
  static_assert(!cuda::std::is_array<volatile T>::value, "");
  static_assert(!cuda::std::is_array<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_array_v<T>, "");
  static_assert(!cuda::std::is_array_v<const T>, "");
  static_assert(!cuda::std::is_array_v<volatile T>, "");
  static_assert(!cuda::std::is_array_v<const volatile T>, "");
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
  test_is_array<char[3]>();
  test_is_array<char[]>();
  test_is_array<Union[]>();

  test_is_not_array<cuda::std::nullptr_t>();
  test_is_not_array<void>();
  test_is_not_array<int&>();
  test_is_not_array<int&&>();
  test_is_not_array<int*>();
  test_is_not_array<double>();
  test_is_not_array<const int*>();
  test_is_not_array<Enum>();
  test_is_not_array<Union>();
  test_is_not_array<FunctionPtr>();
  test_is_not_array<Empty>();
  test_is_not_array<bit_zero>();
  test_is_not_array<NotEmpty>();
  test_is_not_array<incomplete_type>(); //  LWG#2582

  return 0;
}
