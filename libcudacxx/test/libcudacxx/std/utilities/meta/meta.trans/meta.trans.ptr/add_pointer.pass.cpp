//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4.8

// type_traits

// add_pointer
// If T names a referenceable type or a (possibly cv-qualified) void type then
//    the member typedef type shall name the same type as remove_reference_t<T>*;
//    otherwise, type shall name T.

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_add_pointer()
{
  ASSERT_SAME_TYPE(U, typename cuda::std::add_pointer<T>::type);
#if TEST_STD_VER >= 2014
  ASSERT_SAME_TYPE(U, cuda::std::add_pointer_t<T>);
#endif // TEST_STD_VER >= 2014
}

template <class F>
__host__ __device__ void test_function0()
{
  ASSERT_SAME_TYPE(F*, typename cuda::std::add_pointer<F>::type);
#if TEST_STD_VER >= 2014
  ASSERT_SAME_TYPE(F*, cuda::std::add_pointer_t<F>);
#endif // TEST_STD_VER >= 2014
}

template <class F>
__host__ __device__ void test_function1()
{
  ASSERT_SAME_TYPE(F, typename cuda::std::add_pointer<F>::type);
#if TEST_STD_VER >= 2014
  ASSERT_SAME_TYPE(F, cuda::std::add_pointer_t<F>);
#endif // TEST_STD_VER >= 2014
}

struct Foo
{};

int main(int, char**)
{
  test_add_pointer<void, void*>();
  test_add_pointer<int, int*>();
  test_add_pointer<int[3], int(*)[3]>();
  test_add_pointer<int&, int*>();
  test_add_pointer<const int&, const int*>();
  test_add_pointer<int*, int**>();
  test_add_pointer<const int*, const int**>();
  test_add_pointer<Foo, Foo*>();

  //  LWG 2101 specifically talks about add_pointer and functions.
  //  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
  test_function0<void()>();
  test_function1<void() const>();
  test_function1<void()&>();
  test_function1<void() &&>();
  test_function1<void() const&>();
  test_function1<void() const&&>();

  //  But a cv- or ref-qualified member function *is* "a referenceable type"
  test_function0<void (Foo::*)()>();
  test_function0<void (Foo::*)() const>();
  test_function0<void (Foo::*)()&>();
  test_function0<void (Foo::*)() &&>();
  test_function0<void (Foo::*)() const&>();
  test_function0<void (Foo::*)() const&&>();

  return 0;
}
