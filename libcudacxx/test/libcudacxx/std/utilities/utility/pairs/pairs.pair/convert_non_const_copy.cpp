//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class U, class V> EXPLICIT constexpr pair(const pair<U, V>& p);

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

template <class T1, class U1, bool CanCopy = true, bool CanConvert = CanCopy>
TEST_FUNC constexpr void test_pair_non_const()
{
  using P1  = cuda::std::pair<T1, int>;
  using P2  = cuda::std::pair<int, T1>;
  using UP1 = cuda::std::pair<U1, int>&;
  using UP2 = cuda::std::pair<int, U1>&;
  static_assert(cuda::std::is_constructible_v<P1, UP1> == CanCopy);
  static_assert(test_convertible<P1, UP1>() == CanConvert);
  static_assert(cuda::std::is_constructible_v<P2, UP2> == CanCopy);
  static_assert(test_convertible<P2, UP2>() == CanConvert);
}

template <class T, class U>
struct DPair : public cuda::std::pair<T, U>
{
  using Base = cuda::std::pair<T, U>;
  using Base::Base;
};

struct ExplicitT
{
  TEST_FUNC constexpr explicit ExplicitT(int x)
      : value(x)
  {}
  TEST_FUNC constexpr explicit ExplicitT(ExplicitT const& o)
      : value(o.value)
  {}
  int value;
};

struct ImplicitT
{
  TEST_FUNC constexpr ImplicitT(int x)
      : value(x)
  {}
  TEST_FUNC constexpr ImplicitT(ImplicitT const& o)
      : value(o.value)
  {}
  int value;
};

TEST_FUNC constexpr bool test()
{
  {
    using P1 = cuda::std::pair<int, int>;
    using P2 = cuda::std::pair<double, long>;
    P1 p1(3, 4);
    const P2 p2 = p1;
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    // We allow derived types to use this constructor
    using P1 = DPair<long, long>;
    using P2 = cuda::std::pair<int, int>;
    P1 p1(42, 101);
    const P2 p2(p1);
    assert(p2.first == 42);
    assert(p2.second == 101);
  }
  {
    using P1 = cuda::std::pair<int, int>;
    using P2 = cuda::std::pair<double, long>;
    P1 p1(3, 4);
    const P2 p2 = p1;
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    using P1 = cuda::std::pair<int, int>;
    using P2 = cuda::std::pair<ExplicitT, ExplicitT>;
    P1 p1(42, 101);
    const P2 p2(p1);
    assert(p2.first.value == 42);
    assert(p2.second.value == 101);
  }
  {
    using P1 = cuda::std::pair<int, int>;
    using P2 = cuda::std::pair<ImplicitT, ImplicitT>;
    P1 p1(42, 101);
    const P2 p2 = p1;
    assert(p2.first.value == 42);
    assert(p2.second.value == 101);
  }
  {
    test_pair_non_const<AllCtors, AllCtors>(); // copy construction
    test_pair_non_const<AllCtors, AllCtors&>();
    test_pair_non_const<AllCtors, AllCtors&&>();
    test_pair_non_const<AllCtors, const AllCtors&>();
    test_pair_non_const<AllCtors, const AllCtors&&>();

    test_pair_non_const<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors>(); // copy construction
    test_pair_non_const<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&, true, false>();
    test_pair_non_const<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&&, true, false>();
    test_pair_non_const<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&, true, false>();
    test_pair_non_const<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&&, true, false>();

    test_pair_non_const<MoveOnly, MoveOnly, false>(); // copy construction
    test_pair_non_const<MoveOnly, MoveOnly&, false>();
    test_pair_non_const<MoveOnly, MoveOnly&&, false>();

    test_pair_non_const<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly, false>(); // copy construction
    test_pair_non_const<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&, false>();
    test_pair_non_const<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&&, false>();

    test_pair_non_const<CopyOnly, CopyOnly>();
    test_pair_non_const<CopyOnly, CopyOnly&>();
    test_pair_non_const<CopyOnly, CopyOnly&&>();

    test_pair_non_const<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly>();
    test_pair_non_const<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&, true, false>();
    test_pair_non_const<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&&, true, false>();

    test_pair_non_const<NonCopyable, NonCopyable, false>();
    test_pair_non_const<NonCopyable, NonCopyable&, false>();
    test_pair_non_const<NonCopyable, NonCopyable&&, false>();
    test_pair_non_const<NonCopyable, const NonCopyable&, false>();
    test_pair_non_const<NonCopyable, const NonCopyable&&, false>();
  }

  { // Test construction of references
    test_pair_non_const<NonCopyable&, NonCopyable&>();
    test_pair_non_const<NonCopyable&, NonCopyable&&>();
    test_pair_non_const<NonCopyable&, NonCopyable const&, false>();
    test_pair_non_const<NonCopyable const&, NonCopyable&&>();
    test_pair_non_const<NonCopyable&&, NonCopyable&&, false>();

    test_pair_non_const<ConvertingType&, int, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType&, int, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType&&, int, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType const&, int, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType const&&, int, false>();
  }
  {
    test_pair_non_const<AllCtors, int, false>();
    test_pair_non_const<ExplicitTypes::AllCtors, int, false>();
    test_pair_non_const<ConvertingType, int>();
    test_pair_non_const<ExplicitTypes::ConvertingType, int, true, false>();

    test_pair_non_const<ConvertingType, int>();
    test_pair_non_const<ConvertingType, ConvertingType>();
    test_pair_non_const<ConvertingType, ConvertingType const&>();
    test_pair_non_const<ConvertingType, ConvertingType&>();
    test_pair_non_const<ConvertingType, ConvertingType&&>();

    test_pair_non_const<ExplicitTypes::ConvertingType, int, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, int&, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, const int&, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, int&&, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, const int&&, true, false>();

    test_pair_non_const<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType>();
    test_pair_non_const<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType const&, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&, true, false>();
    test_pair_non_const<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&&, true, false>();
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
