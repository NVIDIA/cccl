//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// template <class... Types> class tuple;

// ~tuple();

// C++17 added:
//   The destructor of tuple shall be a trivial destructor
//     if (is_trivially_destructible_v<Types> && ...) is true.

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct not_trivially_destructible
{
  __host__ __device__ virtual ~not_trivially_destructible() {}
};

int main(int, char**)
{
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<>>::value, "");
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<void*>>::value, "");
  static_assert(cuda::std::is_trivially_destructible<cuda::std::tuple<int, float>>::value, "");
  // cuda::std::string is not supported
  /*
  static_assert(!cuda::std::is_trivially_destructible<
      cuda::std::tuple<not_trivially_destructible> >::value, "");
  static_assert(!cuda::std::is_trivially_destructible<
      cuda::std::tuple<int, not_trivially_destructible> >::value, "");
  */
  // non-string check
  static_assert(!cuda::std::is_trivially_destructible<cuda::std::tuple<not_trivially_destructible>>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<cuda::std::tuple<int, not_trivially_destructible>>::value, "");
  return 0;
}
