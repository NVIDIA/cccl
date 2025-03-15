//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// void_t

// XFAIL: gcc-5.1, gcc-5.2

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test1()
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T>>);
}

template <class T, class U>
__host__ __device__ void test2()
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T, U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T, U>>);

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, const T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, volatile T>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<U, const volatile T>>);

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<volatile T, const U>>);
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<const volatile T, const U>>);
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<>>);

  test1<void>();
  test1<int>();
  test1<double>();
  test1<int&>();
  test1<Class>();
  test1<Class[]>();
  test1<Class[5]>();

  test2<void, int>();
  test2<double, int>();
  test2<int&, int>();
  test2<Class&, bool>();
  test2<void*, int&>();

  static_assert(cuda::std::is_same_v<void, cuda::std::void_t<int, double const&, Class, volatile int[], void>>);

  return 0;
}
