//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copyable

// GCC 4.9 and older have no compiler intrinsic support for
// is_trivially_copyable; its corresponding standard library provides no
// implementation.
// XFAIL: gcc-4.8, gcc-4.9

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_copyable()
{
  static_assert(cuda::std::is_trivially_copyable<T>::value, "");
  static_assert(cuda::std::is_trivially_copyable<const T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_trivially_copyable_v<T>, "");
  static_assert(cuda::std::is_trivially_copyable_v<const T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_trivially_copyable()
{
  static_assert(!cuda::std::is_trivially_copyable<T>::value, "");
  static_assert(!cuda::std::is_trivially_copyable<const T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_trivially_copyable_v<T>, "");
  static_assert(!cuda::std::is_trivially_copyable_v<const T>, "");
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
  test_is_trivially_copyable<int>();
  test_is_trivially_copyable<const int>();
  test_is_trivially_copyable<A>();
  test_is_trivially_copyable<const A>();
  test_is_trivially_copyable<C>();

  test_is_not_trivially_copyable<int&>();
  test_is_not_trivially_copyable<const A&>();
  test_is_not_trivially_copyable<B>();

  return 0;
}
