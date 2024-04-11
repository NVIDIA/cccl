//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_swappable

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// IMPORTANT: The include order is part of the test. We want to pick up
// the following definitions in this order:
//   1) is_swappable, is_nothrow_swappable
//   2) iter_swap, swap_ranges
//   3) swap(T (&)[N], T (&)[N])
// This test checks that (1) and (2) see forward declarations
// for (3).
#include <cuda/std/type_traits>
// #include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  // Use a builtin type so we don't get ADL lookup.
  typedef double T[17][29];
  {
    LIBCPP_STATIC_ASSERT(cuda::std::__is_swappable<T>::value, "");
#if TEST_STD_VER > 2011
    static_assert(cuda::std::is_swappable_v<T>, "");
#endif
  }
  {
    T t1 = {};
    T t2 = {};
    cuda::std::iter_swap(t1, t2);
    cuda::std::swap_ranges(t1, t1 + 17, t2);
  }

  return 0;
}
