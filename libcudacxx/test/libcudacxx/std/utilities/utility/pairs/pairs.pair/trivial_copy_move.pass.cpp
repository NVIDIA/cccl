//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// pair(pair const&) = default;
// pair(pair&&) = default;

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct Dummy
{
  Dummy(Dummy const&) = delete;
  Dummy(Dummy&&)      = default;
};

int main(int, char**)
{
  typedef cuda::std::pair<int, short> P;
  {
    static_assert(cuda::std::is_copy_constructible<P>::value, "");
    static_assert(cuda::std::is_trivially_copy_constructible<P>::value, "");
  }
  {
    static_assert(cuda::std::is_move_constructible<P>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<P>::value, "");
  }
  {
    using P1 = cuda::std::pair<Dummy, int>;
    static_assert(!cuda::std::is_copy_constructible<P1>::value, "");
    static_assert(!cuda::std::is_trivially_copy_constructible<P1>::value, "");
    static_assert(cuda::std::is_move_constructible<P1>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<P1>::value, "");
  }

  // extensions to ensure pair is trivially_copyable
  {
    static_assert(cuda::std::is_copy_assignable<P>::value, "");
    static_assert(cuda::std::is_trivially_copy_assignable<P>::value, "");
  }
  {
    static_assert(cuda::std::is_move_assignable<P>::value, "");
    static_assert(cuda::std::is_trivially_move_assignable<P>::value, "");
  }
  {
    static_assert(cuda::std::is_trivially_copyable<P>::value, "");
  }

  return 0;
}
