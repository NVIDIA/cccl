//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_literal_type

// is_literal_type has been deprecated in C++17
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_literal_type()
{
  static_assert(cuda::std::is_literal_type<T>::value, "");
  static_assert(cuda::std::is_literal_type<const T>::value, "");
  static_assert(cuda::std::is_literal_type<volatile T>::value, "");
  static_assert(cuda::std::is_literal_type<const volatile T>::value, "");
  static_assert(cuda::std::is_literal_type_v<T>, "");
  static_assert(cuda::std::is_literal_type_v<const T>, "");
  static_assert(cuda::std::is_literal_type_v<volatile T>, "");
  static_assert(cuda::std::is_literal_type_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_literal_type()
{
  static_assert(!cuda::std::is_literal_type<T>::value, "");
  static_assert(!cuda::std::is_literal_type<const T>::value, "");
  static_assert(!cuda::std::is_literal_type<volatile T>::value, "");
  static_assert(!cuda::std::is_literal_type<const volatile T>::value, "");
  static_assert(!cuda::std::is_literal_type_v<T>, "");
  static_assert(!cuda::std::is_literal_type_v<const T>, "");
  static_assert(!cuda::std::is_literal_type_v<volatile T>, "");
  static_assert(!cuda::std::is_literal_type_v<const volatile T>, "");
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

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_literal_type<cuda::std::nullptr_t>();

  // Before C++14, void was not a literal type
  // In C++14, cv-void is a literal type
  test_is_literal_type<void>();

  test_is_literal_type<int>();
  test_is_literal_type<int*>();
  test_is_literal_type<const int*>();
  test_is_literal_type<int&>();
  test_is_literal_type<int&&>();
  test_is_literal_type<double>();
  test_is_literal_type<char[3]>();
  test_is_literal_type<char[]>();
  test_is_literal_type<Empty>();
  test_is_literal_type<bit_zero>();
  test_is_literal_type<Union>();
  test_is_literal_type<Enum>();
  test_is_literal_type<FunctionPtr>();

  test_is_not_literal_type<NotEmpty>();
  test_is_not_literal_type<Abstract>();

  return 0;
}
