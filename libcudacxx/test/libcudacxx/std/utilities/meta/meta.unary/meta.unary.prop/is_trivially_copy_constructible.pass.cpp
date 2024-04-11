//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copy_constructible

// XFAIL: gcc-4.8, gcc-4.9

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_copy_constructible()
{
  static_assert(cuda::std::is_trivially_copy_constructible<T>::value, "");
  static_assert(cuda::std::is_trivially_copy_constructible<const T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_trivially_copy_constructible_v<T>, "");
  static_assert(cuda::std::is_trivially_copy_constructible_v<const T>, "");
#endif
}

template <class T>
__host__ __device__ void test_has_not_trivial_copy_constructor()
{
  static_assert(!cuda::std::is_trivially_copy_constructible<T>::value, "");
  static_assert(!cuda::std::is_trivially_copy_constructible<const T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_trivially_copy_constructible_v<T>, "");
  static_assert(!cuda::std::is_trivially_copy_constructible_v<const T>, "");
#endif
}

class Empty
{};

class NotEmpty
{
public:
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
public:
  __host__ __device__ virtual ~Abstract() = 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

int main(int, char**)
{
  test_has_not_trivial_copy_constructor<void>();
  test_has_not_trivial_copy_constructor<A>();
  test_has_not_trivial_copy_constructor<Abstract>();
  test_has_not_trivial_copy_constructor<NotEmpty>();

  test_is_trivially_copy_constructible<int&>();
  test_is_trivially_copy_constructible<Union>();
  test_is_trivially_copy_constructible<Empty>();
  test_is_trivially_copy_constructible<int>();
  test_is_trivially_copy_constructible<double>();
  test_is_trivially_copy_constructible<int*>();
  test_is_trivially_copy_constructible<const int*>();
  test_is_trivially_copy_constructible<bit_zero>();

  return 0;
}
