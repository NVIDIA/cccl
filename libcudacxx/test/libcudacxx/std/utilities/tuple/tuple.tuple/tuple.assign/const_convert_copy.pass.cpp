//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... UTypes>
// constexpr const tuple& operator=(const tuple<UTypes...>& u) const;
//
// Constraints:
// - sizeof...(Types) equals sizeof...(UTypes) and
// - (is_assignable_v<const Types&, const UTypes&> && ...) is true.

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "copy_move_types.h"
#include "test_macros.h"

// test constraints

// sizeof...(Types) equals sizeof...(UTypes)
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<int&>&, const cuda::std::tuple<long&>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int&, int&>&, const cuda::std::tuple<long&>&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int&>&, const cuda::std::tuple<long&, long&>&>);

// (is_assignable_v<const Types&, const UTypes&> && ...) is true
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<ConstCopyAssign>>&,
                                         const cuda::std::tuple<ConstCopyAssign>&>);

static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<ConstCopyAssign>, ConstCopyAssign>&,
                                         const cuda::std::tuple<ConstCopyAssign, ConstCopyAssign>&>);

static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<AssignableFrom<ConstCopyAssign>, CopyAssign>&,
                                          const cuda::std::tuple<ConstCopyAssign, CopyAssign>&>);

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1  = 1;
    int i2  = 2;
    long j1 = 3;
    long j2 = 4;
    const cuda::std::tuple<int&, int&> t1{i1, i2};
    const cuda::std::tuple<long&, long&> t2{j1, j2};
    t2 = t1;
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
    const cuda::std::tuple<ConstCopyAssign> t1{1};
    const cuda::std::tuple<AssignableFrom<ConstCopyAssign>> t2{2};
    t2 = t1;
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cuda::std::tuple<TracedAssignment> t1{};
    const cuda::std::tuple<AssignableFrom<TracedAssignment>> t2{};
    t2 = t1;
    assert(cuda::std::get<0>(t2).v.constCopyAssign == 1);
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
