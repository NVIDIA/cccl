//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/type_traits>
// UNSUPPORTED: c++98, c++03, c++11

#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};
struct B
{
public:
  __host__ __device__ operator A()
  {
    return a;
  }
  A a;
};

class C
{};
class D
{
public:
  __host__ __device__ operator C() noexcept
  {
    return c;
  }
  C c;
};

int main(int, char**)
{
  static_assert((cuda::std::is_nothrow_convertible<int, double>::value), "");
  static_assert(!(cuda::std::is_nothrow_convertible<int, char*>::value), "");

  static_assert(!(cuda::std::is_nothrow_convertible<A, B>::value), "");
  static_assert((cuda::std::is_nothrow_convertible<D, C>::value), "");

  static_assert((cuda::std::is_nothrow_convertible_v<int, double>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int, char*>), "");

  static_assert(!(cuda::std::is_nothrow_convertible_v<A, B>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<D, C>), "");

  static_assert((cuda::std::is_nothrow_convertible_v<const void, void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<volatile void, void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<void, const void>), "");
  static_assert((cuda::std::is_nothrow_convertible_v<void, volatile void>), "");

  static_assert(!(cuda::std::is_nothrow_convertible_v<int[], double[]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[], int[]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], int[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], double[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[5], double[10]>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<int[10], A[10]>), "");

  typedef void V();
  typedef int I();
  static_assert(!(cuda::std::is_nothrow_convertible_v<V, V>), "");
  static_assert(!(cuda::std::is_nothrow_convertible_v<V, I>), "");

  return 0;
}
