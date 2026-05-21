//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: Remove once constant_wrapper is exposed.

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::__type
// UNSUPPORTED: gcc-10 || gcc-11

// REQUIRES: !c++17

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=>(L, R) noexcept
//     -> constant_wrapper<(L::value <=> R::value)>
//       { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<(L, R) noexcept -> constant_wrapper<(L::value < R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=(L, R) noexcept -> constant_wrapper<(L::value <= R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator==(L, R) noexcept -> constant_wrapper<(L::value == R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator!=(L, R) noexcept -> constant_wrapper<(L::value != R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>(L, R) noexcept -> constant_wrapper<(L::value > R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>=(L, R) noexcept -> constant_wrapper<(L::value >= R::value)>
//     { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator==(WithOps l, WithOps r)
  {
    return l.value == r.value;
  }
  TEST_FUNC friend constexpr auto operator!=(WithOps l, WithOps r)
  {
    return l.value != r.value;
  }
  TEST_FUNC friend constexpr auto operator<(WithOps l, WithOps r)
  {
    return l.value < r.value;
  }
  TEST_FUNC friend constexpr auto operator<=(WithOps l, WithOps r)
  {
    return l.value <= r.value;
  }
  TEST_FUNC friend constexpr auto operator>=(WithOps l, WithOps r)
  {
    return l.value >= r.value;
  }
  TEST_FUNC friend constexpr auto operator>(WithOps l, WithOps r)
  {
    return l.value > r.value;
  }
  // TEST_FUNC friend constexpr auto operator<=>(WithOps l, WithOps r)
  // {
  //   return l.value <=> r.value;
  // }
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator==(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value == r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator!=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value != r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator<(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value < r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator<=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value <= r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator>=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value >= r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator>(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value > r.value ? 1 : 0};
  }
  // TEST_FUNC friend constexpr auto operator<=>(OpsReturnNonStructural l, OpsReturnNonStructural r)
  // {
  //   return NonStructural{(l.value < r.value) ? -1 : (l.value > r.value) ? 1 : 0};
  // }
};

struct NoOps
{};

template <class L, class R>
concept HasEqual = requires(L l, R r) {
  { l == r };
};

template <class L, class R>
concept HasNotEqual = requires(L l, R r) {
  { l != r };
};

template <class L, class R>
concept HasLess = requires(L l, R r) {
  { l < r };
};

template <class L, class R>
concept HasLessEqual = requires(L l, R r) {
  { l <= r };
};

template <class L, class R>
concept HasGreater = requires(L l, R r) {
  { l > r };
};

template <class L, class R>
concept HasGreaterEqual = requires(L l, R r) {
  { l >= r };
};

template <class L, class R>
concept HasSpaceship = requires(L l, R r) {
  { l <=> r };
};

template <class L, class R>
concept HasNoexceptEqual = requires(L l, R r) {
  { l == r } noexcept;
};

template <class L, class R>
concept HasNoexceptNotEqual = requires(L l, R r) {
  { l != r } noexcept;
};

template <class L, class R>
concept HasNoexceptLess = requires(L l, R r) {
  { l < r } noexcept;
};

template <class L, class R>
concept HasNoexceptLessEqual = requires(L l, R r) {
  { l <= r } noexcept;
};

template <class L, class R>
concept HasNoexceptGreater = requires(L l, R r) {
  { l > r } noexcept;
};

template <class L, class R>
concept HasNoexceptGreaterEqual = requires(L l, R r) {
  { l >= r } noexcept;
};

template <class L, class R>
concept HasNoexceptSpaceship = requires(L l, R r) {
  { l <=> r } noexcept;
};

// Concept checks for int comparisons
static_assert(HasEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNotEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLess<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLessEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasGreater<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasGreaterEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
// static_assert(HasSpaceship<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);

static_assert(HasNoexceptEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptNotEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLess<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLessEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptGreater<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
// static_assert(HasNoexceptSpaceship<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);

// NoOps
static_assert(!HasEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasNotEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLess<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLessEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasGreater<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasGreaterEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
// static_assert(!HasSpaceship<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);

// Concept checks for WithOps comparisons
static_assert(HasNoexceptEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptNotEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLess<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptLessEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptGreater<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
// static_assert(!HasNoexceptSpaceship<cuda::std::__constant_wrapper<WithOps{6}>,
// cuda::std::__constant_wrapper<WithOps{3}>>,
//               "strong_ordering is not a structural type, so the call falls back to runtime implicit conversion and "
//               "operator<=>, which is noexcept(false)");

// clang-format off
// Non-structural types use implicit conversion to underlying type
static_assert(HasEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasNotEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLess<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLessEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreater<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreaterEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);

static_assert(!HasNoexceptEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptNotEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLess<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLessEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreater<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
// clang-format on

TEST_FUNC constexpr bool test()
{
  {
    // int comparisons: 6 vs 3 - returns constant_wrapper<bool_value>
    cuda::std::__constant_wrapper<6> cw6;
    cuda::std::__constant_wrapper<3> cw3;

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) equal = cw6 == cw3;
    static_assert(!static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) not_equal = cw6 != cw3;
    static_assert(static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cw6 < cw3;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less_equal = cw6 <= cw3;
    static_assert(!static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater = cw6 > cw3;
    static_assert(static_cast<bool>(greater));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cw6 >= cw3;
    static_assert(static_cast<bool>(greater_equal));

    // todo: handle spaceship operator
    // strong_ordering is not a structural type
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> cw3;
    // assert(spaceship == cuda::std::strong_ordering::greater);
  }

  {
    // int comparisons: equal values
    cuda::std::__constant_wrapper<3> cw3a;
    cuda::std::__constant_wrapper<3> cw3b;

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) equal = cw3a == cw3b;
    static_assert(static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) not_equal = cw3a != cw3b;
    static_assert(!static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cw3a < cw3b;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) less_equal = cw3a <= cw3b;
    static_assert(static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater = cw3a >= cw3b;
    static_assert(static_cast<bool>(greater));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) greater_cmp = cw3a > cw3b;
    static_assert(!static_cast<bool>(greater_cmp));

    // todo: handle spaceship operator
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw3a <=> cw3b;
    // assert(spaceship == cuda::std::strong_ordering::equal);
  }

  {
    // WithOps comparisons - returns constant_wrapper<bool_value>
    cuda::std::__constant_wrapper<WithOps{6}> cwWithOps6;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3;

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) equal = cwWithOps6 == cwWithOps3;
    static_assert(!static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) not_equal = cwWithOps6 != cwWithOps3;
    static_assert(static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cwWithOps6 < cwWithOps3;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less_equal = cwWithOps6 <= cwWithOps3;
    static_assert(!static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater = cwWithOps6 > cwWithOps3;
    static_assert(static_cast<bool>(greater));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps6 >= cwWithOps3;
    static_assert(static_cast<bool>(greater_equal));

    // todo: handle spaceship operator
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cwWithOps6 <=> cwWithOps3;
    // assert(spaceship == cuda::std::strong_ordering::greater);
  }

  {
    // WithOps comparisons: equal values
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3a;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3b;

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) equal = cwWithOps3a == cwWithOps3b;
    static_assert(static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) not_equal = cwWithOps3a != cwWithOps3b;
    static_assert(!static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cwWithOps3a < cwWithOps3b;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) less_equal = cwWithOps3a <= cwWithOps3b;
    static_assert(static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps3a >= cwWithOps3b;
    static_assert(static_cast<bool>(greater_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) greater = cwWithOps3a > cwWithOps3b;
    static_assert(!static_cast<bool>(greater));

    // todo: handle spaceship operator
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cwWithOps3a <=> cwWithOps3b;
    // assert(spaceship == cuda::std::strong_ordering::equal);
  }

  {
    // Non-structural return types use implicit conversion
    cuda::std::__constant_wrapper<OpsReturnNonStructural{6}> cwOpt6;
    cuda::std::__constant_wrapper<OpsReturnNonStructural{3}> cwOpt3;

    cuda::std::same_as<NonStructural> decltype(auto) equal = cwOpt6 == cwOpt3;
    assert(equal.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) not_equal = cwOpt6 != cwOpt3;
    assert(not_equal.get() == 1);

    cuda::std::same_as<NonStructural> decltype(auto) less = cwOpt6 < cwOpt3;
    assert(less.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) less_equal = cwOpt6 <= cwOpt3;
    assert(less_equal.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) greater = cwOpt6 > cwOpt3;
    assert(greater.get() == 1);

    cuda::std::same_as<NonStructural> decltype(auto) greater_equal = cwOpt6 >= cwOpt3;
    assert(greater_equal.get() == 1);

    // todo: handle spaceship operator
    // cuda::std::same_as<NonStructural> decltype(auto) spaceship = cwOpt6 <=> cwOpt3;
    // assert(spaceship.get() == 1);
  }

  {
    // Mix with runtime param: these operators are not used (built-in operators)
    cuda::std::__constant_wrapper<6> cw6;
    int i = 3;

    cuda::std::same_as<bool> decltype(auto) equal = cw6 == i;
    assert(!equal);

    cuda::std::same_as<bool> decltype(auto) not_equal = cw6 != i;
    assert(not_equal);

    cuda::std::same_as<bool> decltype(auto) less = cw6 < i;
    assert(!less);

    cuda::std::same_as<bool> decltype(auto) less_equal = cw6 <= i;
    assert(!less_equal);

    cuda::std::same_as<bool> decltype(auto) greater = cw6 > i;
    assert(greater);

    cuda::std::same_as<bool> decltype(auto) greater_equal = cw6 >= i;
    assert(greater_equal);

    // todo: handle spaceship operator
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> i;
    // assert(spaceship == cuda::std::strong_ordering::greater);
  }

  {
    // with integral_constant
    cuda::std::__constant_wrapper<6> cw6;
    cuda::std::integral_constant<int, 3> ic3;

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) equal = cw6 == ic3;
    static_assert(!static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) not_equal = cw6 != ic3;
    static_assert(static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cw6 < ic3;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less_equal = cw6 <= ic3;
    static_assert(!static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater = cw6 > ic3;
    static_assert(static_cast<bool>(greater));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cw6 >= ic3;
    static_assert(static_cast<bool>(greater_equal));

    // todo: handle spaceship operator
    // strong_ordering is not a structural type
    // cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> ic3;
    // assert(spaceship == cuda::std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
