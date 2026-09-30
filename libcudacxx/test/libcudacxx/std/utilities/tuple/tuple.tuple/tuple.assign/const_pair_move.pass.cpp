//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template<class U1, class U2>
// constexpr const tuple& operator=(pair<U1, U2>&& u) const;
//
// - sizeof...(Types) is 2,
// - is_assignable_v<const T1&, U1> is true, and
// - is_assignable_v<const T2&, U2> is true

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_macros.h"

// test constraints

// sizeof...(Types) != 2,
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<int&, int&>&, cuda::std::pair<int&, int&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int&>&, cuda::std::pair<int&, int&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int&, int&, int&>&, cuda::std::pair<int&, int&>&&>);

static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign>&,
                                         cuda::std::pair<ConstMoveAssign, ConstMoveAssign>&&>);

// is_assignable_v<const T1&, U1> is false
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<MoveAssign>, ConstMoveAssign>&,
                                          cuda::std::pair<MoveAssign, ConstMoveAssign>&&>);

// is_assignable_v<const T2&, U2> is false
static_assert(
  !cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, AssignableFrom<MoveAssign>>&,
                              cuda::std::pair<ConstMoveAssign, MoveAssign>&&>);

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1  = 1;
    int i2  = 2;
    long j1 = 3;
    long j2 = 4;
    cuda::std::pair<int&, int&> t1{i1, i2};
    const cuda::std::tuple<long&, long&> t2{j1, j2};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2) == 1);
    assert(cuda::std::get<1>(t2) == 2);
    assert(j1 == 1);
    assert(j2 == 2);

    // Ensure the original references have not changed
    assert(cuda::std::addressof(cuda::std::get<0>(t1)) == cuda::std::addressof(i1));
    assert(cuda::std::addressof(cuda::std::get<1>(t1)) == cuda::std::addressof(i2));
    assert(cuda::std::addressof(cuda::std::get<0>(t2)) == cuda::std::addressof(j1));
    assert(cuda::std::addressof(cuda::std::get<1>(t2)) == cuda::std::addressof(j2));
  }

  // user defined const copy assignment
  {
    cuda::std::pair<ConstMoveAssign, ConstMoveAssign> t1{1, 2};
    const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign> t2{3, 4};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).val == 2);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cuda::std::pair<TracedAssignment, TracedAssignment> t1{};
    const cuda::std::tuple<AssignableFrom<TracedAssignment>, AssignableFrom<TracedAssignment>> t2{};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.constMoveAssign == 1);
    assert(cuda::std::get<1>(t2).v.constMoveAssign == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if !TEST_COMPILER(GCC) // gcc cannot have mutable member in constant expression
  static_assert(test());
#endif // !TEST_COMPILER(GCC)
  return 0;
}
