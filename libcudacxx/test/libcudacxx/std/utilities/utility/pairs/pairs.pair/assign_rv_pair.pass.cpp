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

// Copy assignment is non-throwing. Move assignment is not, and it poisons the source.
struct NoexceptCopyAssign
{
  int value = 0;

  TEST_FUNC constexpr NoexceptCopyAssign& operator=(const NoexceptCopyAssign& other) noexcept
  {
    value = other.value;
    return *this;
  }

  TEST_FUNC constexpr NoexceptCopyAssign& operator=(NoexceptCopyAssign&& other)
  {
    value       = other.value;
    other.value = -1;
    return *this;
  }
};

static_assert(cuda::std::is_nothrow_copy_assignable_v<NoexceptCopyAssign>);
static_assert(!cuda::std::is_nothrow_move_assignable_v<NoexceptCopyAssign>);
static_assert(cuda::std::is_nothrow_move_assignable_v<cuda::std::pair<NoexceptCopyAssign&, NoexceptCopyAssign&>>);

// Tracks which assignment operator ran. Copy and move construction are available so a
// pair can store this type by value alongside a reference element.
struct TrackedAssign
{
  int copied = 0;
  int moved  = 0;
  int value  = 0;

  TEST_FUNC constexpr TrackedAssign() {}
  TEST_FUNC constexpr TrackedAssign(int v)
      : value(v)
  {}
  TEST_FUNC constexpr TrackedAssign(const TrackedAssign& other)
      : value(other.value)
  {}
  TEST_FUNC constexpr TrackedAssign(TrackedAssign&& other)
      : value(other.value)
  {}
  TEST_FUNC constexpr TrackedAssign& operator=(const TrackedAssign& other)
  {
    value = other.value;
    ++copied;
    return *this;
  }
  TEST_FUNC constexpr TrackedAssign& operator=(TrackedAssign&& other)
  {
    value = other.value;
    ++moved;
    return *this;
  }
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

    // ... works if it's CopyAssignable. The referents are copy-assigned.
    using P2 = cuda::std::pair<CopyAssignable&, int>;
    static_assert(cuda::std::is_move_assignable<P2>::value);
    CopyAssignable copy_lhs{};
    CopyAssignable copy_rhs{};
    int copy_lhs_second = 1;
    int copy_rhs_second = 2;
    P2 copy_assigned_lhs(copy_lhs, copy_lhs_second);
    P2 copy_assigned_rhs(copy_rhs, copy_rhs_second);
    copy_assigned_lhs = cuda::std::move(copy_assigned_rhs);
    assert(copy_assigned_lhs.second == 2);

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
  { // pair<X&, X&> move assignment forwards the referents as lvalues.
    CountAssign lhs_first{};
    CountAssign lhs_second{};
    CountAssign rhs_first{};
    CountAssign rhs_second{};
    cuda::std::pair<CountAssign&, CountAssign&> lhs(lhs_first, lhs_second);
    cuda::std::pair<CountAssign&, CountAssign&> rhs(rhs_first, rhs_second);
    lhs = cuda::std::move(rhs);
    assert(lhs_first.copied == 1);
    assert(lhs_first.moved == 0);
    assert(lhs_second.copied == 1);
    assert(lhs_second.moved == 0);
    assert(rhs_first.moved == 0);
    assert(rhs_second.moved == 0);
  }
  { // An rvalue-reference element is still move-assigned.
    CountAssign lhs_first{};
    CountAssign lhs_second{};
    CountAssign rhs_first{};
    CountAssign rhs_second{};
    cuda::std::pair<CountAssign&&, CountAssign&&> lhs(cuda::std::move(lhs_first), cuda::std::move(lhs_second));
    cuda::std::pair<CountAssign&&, CountAssign&&> rhs(cuda::std::move(rhs_first), cuda::std::move(rhs_second));
    lhs = cuda::std::move(rhs);
    assert(lhs_first.moved == 1);
    assert(lhs_first.copied == 0);
    assert(lhs_second.moved == 1);
    assert(lhs_second.copied == 0);
  }
  { // Each element is forwarded independently.
    TrackedAssign lhs_ref{1};
    TrackedAssign rhs_ref{2};
    cuda::std::pair<TrackedAssign&, TrackedAssign> lhs(lhs_ref, TrackedAssign{3});
    cuda::std::pair<TrackedAssign&, TrackedAssign> rhs(rhs_ref, TrackedAssign{4});
    lhs = cuda::std::move(rhs);
    assert(lhs_ref.copied == 1);
    assert(lhs_ref.moved == 0);
    assert(lhs_ref.value == 2);
    assert(lhs.second.moved == 1);
    assert(lhs.second.copied == 0);
    assert(lhs.second.value == 4);
    assert(rhs_ref.moved == 0);
    assert(rhs_ref.value == 2);
  }
  {
    NoexceptCopyAssign lhs_first{1};
    NoexceptCopyAssign lhs_second{2};
    NoexceptCopyAssign rhs_first{3};
    NoexceptCopyAssign rhs_second{4};
    cuda::std::pair<NoexceptCopyAssign&, NoexceptCopyAssign&> lhs(lhs_first, lhs_second);
    cuda::std::pair<NoexceptCopyAssign&, NoexceptCopyAssign&> rhs(rhs_first, rhs_second);
    lhs = cuda::std::move(rhs);
    assert(lhs_first.value == 3);
    assert(lhs_second.value == 4);
    assert(rhs_first.value == 3);
    assert(rhs_second.value == 4);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
