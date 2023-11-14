//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

//  These are all constexpr in C++20
// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);


#include <cuda/std/array>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <cuda/std/vector>
#endif
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_comparisons.h"

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
  {
    typedef int T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {1, 2, 3};
    C c2 = {1, 2, 3};
    C c3 = {3, 2, 1};
    C c4 = {1, 2, 1};
    assert(testComparisons6(c1, c2, true, false));
    assert(testComparisons6(c1, c3, false, true));
    assert(testComparisons6(c1, c4, false, false));
  }
  {
    typedef int T;
    typedef cuda::std::array<T, 0> C;
    C c1 = {};
    C c2 = {};
    assert(testComparisons6(c1, c2, true, false));
  }

#if TEST_STD_VER > 17
  {
  constexpr cuda::std::array<int, 3> a1 = {1, 2, 3};
  constexpr cuda::std::array<int, 3> a2 = {2, 3, 4};
  static_assert(testComparisons6(a1, a1, true, false), "");
  static_assert(testComparisons6(a1, a2, false, true), "");
  static_assert(testComparisons6(a2, a1, false, false), "");
  }
#endif

  return 0;
}
