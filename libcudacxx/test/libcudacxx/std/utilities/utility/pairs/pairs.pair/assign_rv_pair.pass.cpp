//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair&& p);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_macros.h"

struct CountAssign
{
  int copied              = 0;
  int moved               = 0;
  constexpr CountAssign() = default;
  TEST_FUNC constexpr CountAssign& operator=(CountAssign const&)
  {
    ++copied;
    return *this;
  }
  TEST_FUNC constexpr CountAssign& operator=(CountAssign&&)
  {
    ++moved;
    return *this;
  }
};

struct NotAssignable
{
  NotAssignable& operator=(NotAssignable const&) = delete;
  NotAssignable& operator=(NotAssignable&&)      = delete;
};

struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

struct CopyAssignable
{
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};

TEST_FUNC constexpr bool test()
{
  {
    typedef cuda::std::pair<ConstexprTestTypes::MoveOnly, int> P;
    P p1(3, 4);
    P p2;
    p2 = cuda::std::move(p1);
    assert(p2.first.value == 3);
    assert(p2.second == 4);
  }
  {
    using P = cuda::std::pair<int&, int&&>;
    int x   = 42;
    int y   = 101;
    int x2  = -1;
    int y2  = 300;
    P p1(x, cuda::std::move(y));
    P p2(x2, cuda::std::move(y2));
    p1 = cuda::std::move(p2);
    assert(p1.first == x2);
    assert(p1.second == y2);
  }
  {
    using P = cuda::std::pair<int, ConstexprTestTypes::DefaultOnly>;
    static_assert(!cuda::std::is_move_assignable<P>::value);
  }
  {
    // The move decays to the copy constructor
    using P = cuda::std::pair<CountAssign, ConstexprTestTypes::CopyOnly>;
    static_assert(cuda::std::is_move_assignable<P>::value);
    P p;
    P p2;
    p = cuda::std::move(p2);
    assert(p.first.moved == 0);
    assert(p.first.copied == 1);
    assert(p2.first.moved == 0);
    assert(p2.first.copied == 0);
  }
  {
    using P = cuda::std::pair<CountAssign, ConstexprTestTypes::MoveOnly>;
    static_assert(cuda::std::is_move_assignable<P>::value);
    P p;
    P p2;
    p = cuda::std::move(p2);
    assert(p.first.moved == 1);
    assert(p.first.copied == 0);
    assert(p2.first.moved == 0);
    assert(p2.first.copied == 0);
  }
  {
    using P1 = cuda::std::pair<int, NotAssignable>;
    using P2 = cuda::std::pair<NotAssignable, int>;
    using P3 = cuda::std::pair<NotAssignable, NotAssignable>;
    static_assert(!cuda::std::is_move_assignable<P1>::value);
    static_assert(!cuda::std::is_move_assignable<P2>::value);
    static_assert(!cuda::std::is_move_assignable<P3>::value);
  }
  {
    // We assign through the reference and don't move out of the incoming ref,
    // so this doesn't work (but would if the type were CopyAssignable).
    using P1 = cuda::std::pair<MoveAssignable&, int>;
    static_assert(!cuda::std::is_move_assignable<P1>::value);

    // ... works if it's CopyAssignable
    using P2 = cuda::std::pair<CopyAssignable&, int>;
    static_assert(cuda::std::is_move_assignable<P2>::value);

    // For rvalue-references, we can move-assign if the type is MoveAssignable
    // or CopyAssignable (since in the worst case the move will decay into a copy).
    using P3 = cuda::std::pair<MoveAssignable&&, int>;
    using P4 = cuda::std::pair<CopyAssignable&&, int>;
    static_assert(cuda::std::is_move_assignable<P3>::value);
    static_assert(cuda::std::is_move_assignable<P4>::value);

    // In all cases, we can't move-assign if the types are not assignable,
    // since we assign through the reference.
    using P5 = cuda::std::pair<NotAssignable&, int>;
    using P6 = cuda::std::pair<NotAssignable&&, int>;
    static_assert(!cuda::std::is_move_assignable<P5>::value);
    static_assert(!cuda::std::is_move_assignable<P6>::value);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
