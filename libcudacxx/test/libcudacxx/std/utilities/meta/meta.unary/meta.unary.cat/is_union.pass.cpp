//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_union

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_union()
{
  static_assert(cuda::std::is_union<T>::value, "");
  static_assert(cuda::std::is_union<const T>::value, "");
  static_assert(cuda::std::is_union<volatile T>::value, "");
  static_assert(cuda::std::is_union<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_union_v<T>, "");
  static_assert(cuda::std::is_union_v<const T>, "");
  static_assert(cuda::std::is_union_v<volatile T>, "");
  static_assert(cuda::std::is_union_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_union()
{
  static_assert(!cuda::std::is_union<T>::value, "");
  static_assert(!cuda::std::is_union<const T>::value, "");
  static_assert(!cuda::std::is_union<volatile T>::value, "");
  static_assert(!cuda::std::is_union<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_union_v<T>, "");
  static_assert(!cuda::std::is_union_v<const T>, "");
  static_assert(!cuda::std::is_union_v<volatile T>, "");
  static_assert(!cuda::std::is_union_v<const volatile T>, "");
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
  test_is_union<Union>();

  test_is_not_union<cuda::std::nullptr_t>();
  test_is_not_union<void>();
  test_is_not_union<int>();
  test_is_not_union<int&>();
  test_is_not_union<int&&>();
  test_is_not_union<int*>();
  test_is_not_union<double>();
  test_is_not_union<const int*>();
  test_is_not_union<char[3]>();
  test_is_not_union<char[]>();
  test_is_not_union<Enum>();
  test_is_not_union<FunctionPtr>();
  test_is_not_union<Empty>();
  test_is_not_union<bit_zero>();
  test_is_not_union<NotEmpty>();
  test_is_not_union<Abstract>();
  test_is_not_union<incomplete_type>();

  return 0;
}
