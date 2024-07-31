//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// void_t

// UNSUPPORTED: c++98, c++03

// XFAIL: gcc-5.1, gcc-5.2

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test1()
{
  ASSERT_SAME_TYPE(void, cuda::std::void_t<T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<volatile T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const volatile T>);
}

template <class T, class U>
__host__ __device__ void test2()
{
  ASSERT_SAME_TYPE(void, cuda::std::void_t<T, U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const T, U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<volatile T, U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const volatile T, U>);

  ASSERT_SAME_TYPE(void, cuda::std::void_t<U, T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<U, const T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<U, volatile T>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<U, const volatile T>);

  ASSERT_SAME_TYPE(void, cuda::std::void_t<T, const U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const T, const U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<volatile T, const U>);
  ASSERT_SAME_TYPE(void, cuda::std::void_t<const volatile T, const U>);
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  ASSERT_SAME_TYPE(void, cuda::std::void_t<>);

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

  ASSERT_SAME_TYPE(void, cuda::std::void_t<int, double const&, Class, volatile int[], void>);

  return 0;
}
