//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copyable for volatile qualified types

// These compilers have not implemented Core 2094 which makes volatile
// qualified types trivially copyable.
// XFAIL: clang-3, clang-4, apple-clang-6, apple-clang-7, apple-clang-8, apple-clang-9.0
// XFAIL: gcc-4.8, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9
// XFAIL: icc

// When we marked this XFAIL for MSVC, QA reported that it unexpectedly passed.
// When we stopped marking it XFAIL for MSVC, QA reported that it unexpectedly
// failed
// UNSUPPORTED: msvc

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_copyable_volatile()
{
  static_assert(cuda::std::is_trivially_copyable<volatile T>::value, "");
  static_assert(cuda::std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_trivially_copyable_v<volatile T>, "");
  static_assert(cuda::std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_trivially_copyable_volatile()
{
  static_assert(!cuda::std::is_trivially_copyable<volatile T>::value, "");
  static_assert(!cuda::std::is_trivially_copyable<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_trivially_copyable_v<volatile T>, "");
  static_assert(!cuda::std::is_trivially_copyable_v<const volatile T>, "");
#endif
}

struct A
{
  int i_;
};

struct B
{
  int i_;
  __host__ __device__ ~B()
  {
    assert(i_ == 0);
  }
};

class C
{
public:
  __host__ __device__ C();
};

int main(int, char**)
{
  test_is_trivially_copyable_volatile<int>();
  test_is_trivially_copyable_volatile<const int>();
  test_is_trivially_copyable_volatile<A>();
  test_is_trivially_copyable_volatile<const A>();
  test_is_trivially_copyable_volatile<C>();

  test_is_not_trivially_copyable_volatile<int&>();
  test_is_not_trivially_copyable_volatile<const A&>();
  test_is_not_trivially_copyable_volatile<B>();

  return 0;
}
