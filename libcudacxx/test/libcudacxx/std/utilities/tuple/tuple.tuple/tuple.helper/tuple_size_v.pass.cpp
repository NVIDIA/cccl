//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: gcc-6

// <cuda/std/tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <cuda/std/tuple>
#include <cuda/std/utility>
// cuda::std::array not supported
// #include <cuda/std/array>

#include "test_macros.h"

template <class Tuple, int Expect>
__host__ __device__ void test()
{
  static_assert(cuda::std::tuple_size_v<Tuple> == Expect, "");
  static_assert(cuda::std::tuple_size_v<Tuple> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple const> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple volatile> == cuda::std::tuple_size<Tuple>::value, "");
  static_assert(cuda::std::tuple_size_v<Tuple const volatile> == cuda::std::tuple_size<Tuple>::value, "");
}

int main(int, char**)
{
  test<cuda::std::tuple<>, 0>();

  test<cuda::std::tuple<int>, 1>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 1>, 1>();

  test<cuda::std::tuple<int, int>, 2>();
  test<cuda::std::pair<int, int>, 2>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 2>, 2>();

  test<cuda::std::tuple<int, int, int>, 3>();
  // cuda::std::array not supported
  // test<cuda::std::array<int, 3>, 3>();

  return 0;
}
