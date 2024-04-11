//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// ~pair()

// C++17 added:
//   The destructor of pair shall be a trivial destructor
//     if (is_trivially_destructible_v<T1> && is_trivially_destructible_v<T2>) is true.

#include <cuda/std/type_traits>
#include <cuda/std/utility>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "DefaultOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  static_assert((cuda::std::is_trivially_destructible<cuda::std::pair<int, float>>::value), "");
  /*
  static_assert((!cuda::std::is_trivially_destructible<
      cuda::std::pair<int, cuda::std::string> >::value), "");
  */
  static_assert((!cuda::std::is_trivially_destructible<cuda::std::pair<int, DefaultOnly>>::value), "");

  return 0;
}
