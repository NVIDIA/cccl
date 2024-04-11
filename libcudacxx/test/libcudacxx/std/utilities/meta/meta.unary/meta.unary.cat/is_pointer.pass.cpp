//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_pointer

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_pointer()
{
  static_assert(cuda::std::is_pointer<T>::value, "");
  static_assert(cuda::std::is_pointer<const T>::value, "");
  static_assert(cuda::std::is_pointer<volatile T>::value, "");
  static_assert(cuda::std::is_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_pointer_v<T>, "");
  static_assert(cuda::std::is_pointer_v<const T>, "");
  static_assert(cuda::std::is_pointer_v<volatile T>, "");
  static_assert(cuda::std::is_pointer_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_pointer()
{
  static_assert(!cuda::std::is_pointer<T>::value, "");
  static_assert(!cuda::std::is_pointer<const T>::value, "");
  static_assert(!cuda::std::is_pointer<volatile T>::value, "");
  static_assert(!cuda::std::is_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_pointer_v<T>, "");
  static_assert(!cuda::std::is_pointer_v<const T>, "");
  static_assert(!cuda::std::is_pointer_v<volatile T>, "");
  static_assert(!cuda::std::is_pointer_v<const volatile T>, "");
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
  test_is_pointer<void*>();
  test_is_pointer<int*>();
  test_is_pointer<const int*>();
  test_is_pointer<Abstract*>();
  test_is_pointer<FunctionPtr>();

  test_is_not_pointer<cuda::std::nullptr_t>();
  test_is_not_pointer<void>();
  test_is_not_pointer<int&>();
  test_is_not_pointer<int&&>();
  test_is_not_pointer<double>();
  test_is_not_pointer<char[3]>();
  test_is_not_pointer<char[]>();
  test_is_not_pointer<Union>();
  test_is_not_pointer<Enum>();
  test_is_not_pointer<Empty>();
  test_is_not_pointer<bit_zero>();
  test_is_not_pointer<NotEmpty>();
  test_is_not_pointer<Abstract>();
  test_is_not_pointer<incomplete_type>();

  return 0;
}
