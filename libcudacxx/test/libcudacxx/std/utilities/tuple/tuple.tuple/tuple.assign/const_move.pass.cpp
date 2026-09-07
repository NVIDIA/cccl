//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <tuple>

// constexpr const tuple& operator=(tuple&&) const;
//
// Constraints: (is_assignable_v<const Types&, Types> && ...) is true.

// test constraints

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "copy_move_types.h"
#include "test_macros.h"

static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int>&, cuda::std::tuple<int>&&>);
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<int&>&, cuda::std::tuple<int&>&&>);
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<int&, int&>&, cuda::std::tuple<int&, int&>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<int&, int>&, cuda::std::tuple<int&, int>&&>);

// this is fallback to tuple's const copy assignment
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<ConstCopyAssign>&, cuda::std::tuple<ConstCopyAssign>&&>);

static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<CopyAssign>&, cuda::std::tuple<CopyAssign>&&>);
static_assert(cuda::std::is_assignable_v<const cuda::std::tuple<ConstMoveAssign>&, cuda::std::tuple<ConstMoveAssign>&&>);
static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<MoveAssign>&, cuda::std::tuple<MoveAssign>&&>);

// Ensure empty tuple is also trivially assignable, which implies its not const assignable
static_assert(!cuda::std::is_move_assignable_v<const cuda::std::tuple<>>);
static_assert(cuda::std::is_trivially_move_assignable_v<cuda::std::tuple<>>);

TEST_FUNC constexpr bool test()
{
  // reference types
  {
    int i1    = 1;
    int i2    = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    cuda::std::tuple<int&, double&> t1{i1, d1};
    const cuda::std::tuple<int&, double&> t2{i2, d2};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2) == 1);
    assert(cuda::std::get<1>(t2) == 3.0);
    assert(i2 == 1);
    assert(d2 == 3.0);

    // Ensure the original references have not changed
    assert(cuda::std::addressof(cuda::std::get<0>(t1)) == cuda::std::addressof(i1));
    assert(cuda::std::addressof(cuda::std::get<1>(t1)) == cuda::std::addressof(d1));
    assert(cuda::std::addressof(cuda::std::get<0>(t2)) == cuda::std::addressof(i2));
    assert(cuda::std::addressof(cuda::std::get<1>(t2)) == cuda::std::addressof(d2));
  }

  // user defined const move assignment
  {
    cuda::std::tuple<ConstMoveAssign> t1{1};
    const cuda::std::tuple<ConstMoveAssign> t2{2};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).val == 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cuda::std::tuple<TracedAssignment, const TracedAssignment> t1{};
    const cuda::std::tuple<TracedAssignment, const TracedAssignment> t2{};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).constMoveAssign == 1);
    assert(cuda::std::get<1>(t2).constCopyAssign == 1);
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
