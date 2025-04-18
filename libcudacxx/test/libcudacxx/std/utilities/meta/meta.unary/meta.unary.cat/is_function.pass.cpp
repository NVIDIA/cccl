//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_function

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

// NOTE: On Windows the function `test_is_function<void()>` and
// `test_is_function<void() noexcept> has the same mangled despite being
// a distinct instantiation. This causes Clang to emit an error. However
// structs do not have this problem.

template <class T>
struct test_is_function
{
  static_assert(cuda::std::is_function<T>::value, "");
  static_assert(cuda::std::is_function<const T>::value, "");
  static_assert(cuda::std::is_function<volatile T>::value, "");
  static_assert(cuda::std::is_function<const volatile T>::value, "");
  static_assert(cuda::std::is_function_v<T>, "");
  static_assert(cuda::std::is_function_v<const T>, "");
  static_assert(cuda::std::is_function_v<volatile T>, "");
  static_assert(cuda::std::is_function_v<const volatile T>, "");
};

template <class T>
struct test_is_not_function
{
  static_assert(!cuda::std::is_function<T>::value, "");
  static_assert(!cuda::std::is_function<const T>::value, "");
  static_assert(!cuda::std::is_function<volatile T>::value, "");
  static_assert(!cuda::std::is_function<const volatile T>::value, "");
  static_assert(!cuda::std::is_function_v<T>, "");
  static_assert(!cuda::std::is_function_v<const T>, "");
  static_assert(!cuda::std::is_function_v<volatile T>, "");
  static_assert(!cuda::std::is_function_v<const volatile T>, "");
};

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
  test_is_function<void()>();
  test_is_function<int(int)>();
  test_is_function<int(int, double)>();
  test_is_function<int(Abstract*)>();
  test_is_function<void(...)>();

  test_is_not_function<cuda::std::nullptr_t>();
  test_is_not_function<void>();
  test_is_not_function<int>();
  test_is_not_function<int&>();
  test_is_not_function<int&&>();
  test_is_not_function<int*>();
  test_is_not_function<double>();
  test_is_not_function<char[3]>();
  test_is_not_function<char[]>();
  test_is_not_function<Union>();
  test_is_not_function<Enum>();
  test_is_not_function<FunctionPtr>(); // function pointer is not a function
  test_is_not_function<Empty>();
  test_is_not_function<bit_zero>();
  test_is_not_function<NotEmpty>();
  test_is_not_function<Abstract>();
  test_is_not_function<Abstract*>();
  test_is_not_function<incomplete_type>();

  test_is_function<void() noexcept>();
  test_is_function<void() const && noexcept>();

  return 0;
}
