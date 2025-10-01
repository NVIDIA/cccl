//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Tuple, __tuple_convertible<Tuple, tuple> >
//   tuple(Tuple &&);
//
// template <class Tuple, __tuple_constructible<Tuple, tuple> >
//   tuple(Tuple &&);

// This test checks that we do not evaluate __make_tuple_types
// on the array.

// cuda::std::array not supported
// #include <cuda/std/array>
#include <cuda/std/tuple>

#include "test_macros.h"

// cuda::std::array not supported
/*
// Use 1256 to try and blow the template instantiation depth for all compilers.
using array_t = cuda::std::array<char, 1256>;
using tuple_t = cuda::std::tuple<array_t>;
*/

int main(int, char**)
{
  /*
  array_t arr;
  tuple_t tup(arr);
  */
  return 0;
}
