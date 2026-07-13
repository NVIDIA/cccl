//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair
// constexpr const pair& operator=(pair&& p) const;

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_macros.h"

// Constraints:
// is_assignable<const first_type&, first_type> is true and
// is_assignable<const second_type&, second_type> is true.

// clang-format off
static_assert(cuda::std::is_assignable_v<const cuda::std::pair<int&&, int&&>&,
                                   cuda::std::pair<int&&, int&&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int>&,
                                    cuda::std::pair<int, int>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int&&>&,
                                    cuda::std::pair<int, int&&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int&&, int>&,
                                    cuda::std::pair<int&&, int>&&>);

static_assert(cuda::std::is_assignable_v<const cuda::std::pair<ConstMoveAssign, ConstMoveAssign>&,
                                   cuda::std::pair<ConstMoveAssign, ConstMoveAssign>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<MoveAssign, MoveAssign>&,
                                   cuda::std::pair<MoveAssign, MoveAssign>&&>);

// clang-format on

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1    = 1;
    int i2    = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    cuda::std::pair<int&&, double&&> p1{cuda::std::move(i1), cuda::std::move(d1)};
    const cuda::std::pair<int&&, double&&> p2{cuda::std::move(i2), cuda::std::move(d2)};
    p2 = cuda::std::move(p1);
    assert(p2.first == 1);
    assert(p2.second == 3.0);
  }

  // user defined const move assignment
  {
    cuda::std::pair<ConstMoveAssign, ConstMoveAssign> p1{1, 2};
    const cuda::std::pair<ConstMoveAssign, ConstMoveAssign> p2{3, 4};
    p2 = cuda::std::move(p1);
    assert(p2.first.val == 1);
    assert(p2.second.val == 2);
  }

  // The correct assignment operator of the underlying type is used
  {
    cuda::std::pair<TracedAssignment, const TracedAssignment> t1{};
    const cuda::std::pair<TracedAssignment, const TracedAssignment> t2{};
    t2 = cuda::std::move(t1);
    assert(t2.first.constMoveAssign == 1);
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
