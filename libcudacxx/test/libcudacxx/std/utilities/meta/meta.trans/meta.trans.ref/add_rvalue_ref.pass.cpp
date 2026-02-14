//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4.8

// type_traits

// add_rvalue_reference
// If T names a referenceable type then the member typedef type
//   shall name T&&; otherwise, type shall name T.

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_add_rvalue_reference()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::add_rvalue_reference<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::add_rvalue_reference_t<T>>);
}

template <class F>
__host__ __device__ void test_function0()
{
  static_assert(cuda::std::is_same_v<F&&, typename cuda::std::add_rvalue_reference<F>::type>);
  static_assert(cuda::std::is_same_v<F&&, cuda::std::add_rvalue_reference_t<F>>);
}

template <class F>
__host__ __device__ void test_function1()
{
  static_assert(cuda::std::is_same_v<F, typename cuda::std::add_rvalue_reference<F>::type>);
  static_assert(cuda::std::is_same_v<F, cuda::std::add_rvalue_reference_t<F>>);
}

struct Foo
{};

int main(int, char**)
{
  test_add_rvalue_reference<void, void>();
  test_add_rvalue_reference<int, int&&>();
  test_add_rvalue_reference<int[3], int (&&)[3]>();
  test_add_rvalue_reference<int&, int&>();
  test_add_rvalue_reference<const int&, const int&>();
  test_add_rvalue_reference<int*, int*&&>();
  test_add_rvalue_reference<const int*, const int*&&>();
  test_add_rvalue_reference<Foo, Foo&&>();

  //  LWG 2101 specifically talks about add_rvalue_reference and functions.
  //  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
  test_function0<void()>();
  test_function1<void() const>();
  test_function1<void() &>();
  test_function1<void() &&>();
  test_function1<void() const&>();
  test_function1<void() const&&>();

  //  But a cv- or ref-qualified member function *is* "a referenceable type"
  test_function0<void (Foo::*)()>();
  test_function0<void (Foo::*)() const>();
  test_function0<void (Foo::*)() &>();
  test_function0<void (Foo::*)() &&>();
  test_function0<void (Foo::*)() const&>();
  test_function0<void (Foo::*)() const&&>();

  return 0;
}
