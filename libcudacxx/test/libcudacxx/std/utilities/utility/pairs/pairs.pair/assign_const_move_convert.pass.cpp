//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair
// template <class U1, class U2>
// constexpr const pair& operator=(pair<U1, U2>&& p) const;

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_macros.h"

// Constraints:
// is_assignable<const first_type&, U1> is true and
// is_assignable<const second_type&, U2> is true.

// clang-format off
static_assert( cuda::std::is_assignable_v<const cuda::std::pair<int&&, int&&>&,
                                    cuda::std::pair<long&&, long&&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int>&,
                                    cuda::std::pair<long, long>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int, int&&>&,
                                    cuda::std::pair<long, long&&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::pair<int&&, int>&,
                                    cuda::std::pair<long&&, long>&&>);

static_assert(cuda::std::is_assignable_v<
    const cuda::std::pair<AssignableFrom<ConstMoveAssign>, AssignableFrom<ConstMoveAssign>>&,
    cuda::std::pair<ConstMoveAssign, ConstMoveAssign>&&>);

static_assert(!cuda::std::is_assignable_v<
    const cuda::std::pair<AssignableFrom<MoveAssign>, AssignableFrom<MoveAssign>>&,
    cuda::std::pair<MoveAssign, MoveAssign>&&>);
// clang-format on

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1  = 1;
    int i2  = 2;
    long j1 = 3;
    long j2 = 4;
    cuda::std::pair<int&&, int&&> p1{cuda::std::move(i1), cuda::std::move(i2)};
    const cuda::std::pair<long&&, long&&> p2{cuda::std::move(j1), cuda::std::move(j2)};
    p2 = cuda::std::move(p1);
    assert(p2.first == 1);
    assert(p2.second == 2);
  }

  // user defined const move assignment
  {
    cuda::std::pair<ConstMoveAssign, ConstMoveAssign> p1{1, 2};
    const cuda::std::pair<AssignableFrom<ConstMoveAssign>, AssignableFrom<ConstMoveAssign>> p2{3, 4};
    p2 = cuda::std::move(p1);
    assert(p2.first.v.val == 1);
    assert(p2.second.v.val == 2);
  }

  // The correct assignment operator of the underlying type is used
  {
    cuda::std::pair<TracedAssignment, TracedAssignment> t1{};
    const cuda::std::pair<AssignableFrom<TracedAssignment>, AssignableFrom<TracedAssignment>> t2{};
    t2 = cuda::std::move(t1);
    assert(t2.first.v.constMoveAssign == 1);
    assert(t2.second.v.constMoveAssign == 1);
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
