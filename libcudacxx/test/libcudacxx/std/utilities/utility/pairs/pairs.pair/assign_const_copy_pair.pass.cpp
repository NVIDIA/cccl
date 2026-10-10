//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair
// constexpr const pair& operator=(const pair& p) const;

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_macros.h"

// Constraints:
// is_copy_assignable<const first_type> is true and
// is_copy_assignable<const second_type> is true.

static_assert(cuda::std::is_assignable_v<const cuda::std::pair<int&, int&>&, const cuda::std::pair<int&, int&>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int>&, const cuda::std::pair<int, int>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int&>&, const cuda::std::pair<int, int&>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int&, int>&, const cuda::std::pair<int&, int>&>);

static_assert(cuda::std::is_assignable_v<const cuda::std::pair<ConstCopyAssign, ConstCopyAssign>&,
                                         const cuda::std::pair<ConstCopyAssign, ConstCopyAssign>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<CopyAssign, CopyAssign>&,
                                          const cuda::std::pair<CopyAssign, CopyAssign>&>);

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1    = 1;
    int i2    = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    const cuda::std::pair<int&, double&> p1{i1, d1};
    const cuda::std::pair<int&, double&> p2{i2, d2};
    p2 = p1;
    assert(p2.first == 1);
    assert(p2.second == 3.0);
  }

  // user defined const copy assignment
  {
    const cuda::std::pair<ConstCopyAssign, ConstCopyAssign> p1{1, 2};
    const cuda::std::pair<ConstCopyAssign, ConstCopyAssign> p2{3, 4};
    p2 = p1;
    assert(p2.first.val == 1);
    assert(p2.second.val == 2);
  }

  // The correct assignment operator of the underlying type is used
  {
    cuda::std::pair<TracedAssignment, const TracedAssignment> t1{};
    const cuda::std::pair<TracedAssignment, const TracedAssignment> t2{};
    t2 = t1;
    assert(t2.first.constCopyAssign == 1);
    assert(t2.second.constCopyAssign == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
// gcc cannot have mutable member in constant expression
#if !TEST_COMPILER(GCC)
  static_assert(test());
#endif // !TEST_COMPILER(GCC)
  return 0;
}
