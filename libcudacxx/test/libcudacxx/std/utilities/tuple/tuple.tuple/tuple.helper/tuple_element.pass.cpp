//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, cuda::std::size_t N, class U>
__host__ __device__ void test()
{
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, T>::type, U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, const T>::type, const U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, volatile T>::type, volatile U>::value), "");
  static_assert(
    (cuda::std::is_same<typename cuda::std::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, T>, U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, const T>, const U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, volatile T>, volatile U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, const volatile T>, const volatile U>::value),
                "");
#endif
}

int main(int, char**)
{
  test<cuda::std::tuple<int>, 0, int>();
  test<cuda::std::tuple<char, int>, 0, char>();
  test<cuda::std::tuple<char, int>, 1, int>();
  test<cuda::std::tuple<int*, char, int>, 0, int*>();
  test<cuda::std::tuple<int*, char, int>, 1, char>();
  test<cuda::std::tuple<int*, char, int>, 2, int>();

  return 0;
}
