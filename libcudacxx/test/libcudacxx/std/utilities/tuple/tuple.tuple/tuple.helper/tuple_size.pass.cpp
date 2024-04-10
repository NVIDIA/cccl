//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, cuda::std::size_t N>
__host__ __device__ void test()
{
  static_assert(
    (cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>, cuda::std::tuple_size<T>>::value), "");
  static_assert(
    (cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>, cuda::std::tuple_size<const T>>::value),
    "");
  static_assert(
    (cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>, cuda::std::tuple_size<volatile T>>::value),
    "");
  static_assert((cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>,
                                       cuda::std::tuple_size<const volatile T>>::value),
                "");
}

int main(int, char**)
{
  test<cuda::std::tuple<>, 0>();
  test<cuda::std::tuple<int>, 1>();
  test<cuda::std::tuple<char, int>, 2>();
  test<cuda::std::tuple<char, char*, int>, 3>();

  return 0;
}
