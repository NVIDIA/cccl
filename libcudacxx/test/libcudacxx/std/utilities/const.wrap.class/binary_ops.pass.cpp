//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// nvcc 12.0 segfaults.
// UNSUPPORTED: nvcc-12.0

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator+(L, R) noexcept -> constant_wrapper<(L::value + R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator-(L, R) noexcept -> constant_wrapper<(L::value - R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator*(L, R) noexcept -> constant_wrapper<(L::value * R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator/(L, R) noexcept -> constant_wrapper<(L::value / R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator%(L, R) noexcept -> constant_wrapper<(L::value % R::value)>
//      { return {}; }

//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator<<(L, R) noexcept -> constant_wrapper<(L::value << R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator>>(L, R) noexcept -> constant_wrapper<(L::value >> R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator&(L, R) noexcept -> constant_wrapper<(L::value & R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator|(L, R) noexcept -> constant_wrapper<(L::value | R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator^(L, R) noexcept -> constant_wrapper<(L::value ^ R::value)>
//      { return {}; }

//  template<constexpr-param L, constexpr-param R>
//    requires (!is_constructible_v<bool, decltype(L::value)> ||
//              !is_constructible_v<bool, decltype(R::value)>)
//      friend constexpr auto operator&&(L, R) noexcept
//        -> constant_wrapper<(L::value && R::value)>
//          { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    requires (!is_constructible_v<bool, decltype(L::value)> ||
//              !is_constructible_v<bool, decltype(R::value)>)
//      friend constexpr auto operator||(L, R) noexcept
//        -> constant_wrapper<(L::value || R::value)>
//          { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function
TEST_DIAG_SUPPRESS_CLANG("-Wconstant-logical-operand")

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator+(WithOps l, WithOps r)
  {
    return WithOps{l.value + r.value};
  }
  TEST_FUNC friend constexpr auto operator-(WithOps l, WithOps r)
  {
    return WithOps{l.value - r.value};
  }
  TEST_FUNC friend constexpr auto operator*(WithOps l, WithOps r)
  {
    return WithOps{l.value * r.value};
  }
  TEST_FUNC friend constexpr auto operator/(WithOps l, WithOps r)
  {
    return WithOps{l.value / r.value};
  }
  TEST_FUNC friend constexpr auto operator%(WithOps l, WithOps r)
  {
    return WithOps{l.value % r.value};
  }
  TEST_FUNC friend constexpr auto operator<<(WithOps l, WithOps r)
  {
    return WithOps{l.value << r.value};
  }
  TEST_FUNC friend constexpr auto operator>>(WithOps l, WithOps r)
  {
    return WithOps{l.value >> r.value};
  }
  TEST_FUNC friend constexpr auto operator&(WithOps l, WithOps r)
  {
    return WithOps{l.value & r.value};
  }
  TEST_FUNC friend constexpr auto operator|(WithOps l, WithOps r)
  {
    return WithOps{l.value | r.value};
  }
  TEST_FUNC friend constexpr auto operator^(WithOps l, WithOps r)
  {
    return WithOps{l.value ^ r.value};
  }

  TEST_FUNC friend constexpr auto operator&&(WithOps l, WithOps r)
  {
    return WithOps{l.value && r.value};
  }
  TEST_FUNC friend constexpr auto operator||(WithOps l, WithOps r)
  {
    return WithOps{l.value || r.value};
  }
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator+(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value + r.value};
  }
  TEST_FUNC friend constexpr auto operator-(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value - r.value};
  }
  TEST_FUNC friend constexpr auto operator*(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value * r.value};
  }
  TEST_FUNC friend constexpr auto operator/(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value / r.value};
  }
  TEST_FUNC friend constexpr auto operator%(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value % r.value};
  }
  TEST_FUNC friend constexpr auto operator<<(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value << r.value};
  }
  TEST_FUNC friend constexpr auto operator>>(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value >> r.value};
  }
  TEST_FUNC friend constexpr auto operator&(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value & r.value};
  }
  TEST_FUNC friend constexpr auto operator|(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value | r.value};
  }
  TEST_FUNC friend constexpr auto operator^(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value ^ r.value};
  }
  TEST_FUNC friend constexpr auto operator&&(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value && r.value};
  }
  TEST_FUNC friend constexpr auto operator||(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value || r.value};
  }
};

struct NoOps
{};

template <class L, class R>
concept HasPlus = requires(L l, R r) {
  { l + r };
};

template <class L, class R>
concept HasMinus = requires(L l, R r) {
  { l - r };
};

template <class L, class R>
concept HasMultiply = requires(L l, R r) {
  { l * r };
};

template <class L, class R>
concept HasDivide = requires(L l, R r) {
  { l / r };
};

template <class L, class R>
concept HasModulo = requires(L l, R r) {
  { l % r };
};

template <class L, class R>
concept HasShiftLeft = requires(L l, R r) {
  { l << r };
};

template <class L, class R>
concept HasShiftRight = requires(L l, R r) {
  { l >> r };
};

template <class L, class R>
concept HasBitAnd = requires(L l, R r) {
  { l & r };
};

template <class L, class R>
concept HasBitOr = requires(L l, R r) {
  { l | r };
};

template <class L, class R>
concept HasBitXor = requires(L l, R r) {
  { l ^ r };
};

template <class L, class R>
concept HasLogicalAnd = requires(L l, R r) {
  { l && r };
};

template <class L, class R>
concept HasLogicalOr = requires(L l, R r) {
  { l || r };
};

template <class L, class R>
concept HasNoexceptPlus = requires(L l, R r) {
  { l + r } noexcept;
};

template <class L, class R>
concept HasNoexceptMinus = requires(L l, R r) {
  { l - r } noexcept;
};

template <class L, class R>
concept HasNoexceptMultiply = requires(L l, R r) {
  { l * r } noexcept;
};

template <class L, class R>
concept HasNoexceptDivide = requires(L l, R r) {
  { l / r } noexcept;
};

template <class L, class R>
concept HasNoexceptModulo = requires(L l, R r) {
  { l % r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftLeft = requires(L l, R r) {
  { l << r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftRight = requires(L l, R r) {
  { l >> r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitAnd = requires(L l, R r) {
  { l & r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitOr = requires(L l, R r) {
  { l | r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitXor = requires(L l, R r) {
  { l ^ r } noexcept;
};

template <class L, class R>
concept HasNoexceptLogicalAnd = requires(L l, R r) {
  { l && r } noexcept;
};

template <class L, class R>
concept HasNoexceptLogicalOr = requires(L l, R r) {
  { l || r } noexcept;
};

// Concept checks for int + int operations
static_assert(HasPlus<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasMinus<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasMultiply<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasDivide<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasModulo<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasShiftLeft<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);
static_assert(HasShiftRight<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);
static_assert(HasBitAnd<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasBitOr<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasBitXor<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLogicalAnd<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLogicalOr<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);

static_assert(HasNoexceptPlus<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptMinus<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptMultiply<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptDivide<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptModulo<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptShiftLeft<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);
static_assert(HasNoexceptShiftRight<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);
static_assert(HasNoexceptBitAnd<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptBitOr<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptBitXor<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLogicalAnd<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLogicalOr<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);

// NoOps
static_assert(!HasPlus<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasMinus<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasMultiply<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasDivide<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasModulo<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasShiftLeft<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasShiftRight<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitAnd<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitOr<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitXor<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLogicalAnd<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLogicalOr<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);

// Concept checks for WithOps operations
static_assert(HasNoexceptPlus<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMinus<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMultiply<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptDivide<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptModulo<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptShiftLeft<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{1}>>);
static_assert(
  HasNoexceptShiftRight<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{1}>>);
static_assert(HasNoexceptBitAnd<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitOr<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitXor<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptLogicalAnd<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptLogicalOr<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);

// clang-format off
// Non-structural types use implicit conversion to underlying type
static_assert(HasPlus<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasMinus<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasMultiply<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasDivide<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasModulo<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasShiftLeft<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(HasShiftRight<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(HasBitAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasBitOr<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasBitXor<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLogicalAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLogicalOr<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);

static_assert(!HasNoexceptPlus<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptMinus<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptMultiply<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptDivide<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptModulo<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptShiftLeft<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasNoexceptShiftRight<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasNoexceptBitAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptBitOr<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptBitXor<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLogicalAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLogicalOr<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
// clang-format on

TEST_FUNC constexpr bool test()
{
  {
    // int + int
    cuda::std::__constant_wrapper<6> cw6;
    cuda::std::__constant_wrapper<3> cw3;

    cuda::std::same_as<cuda::std::__constant_wrapper<9>> decltype(auto) result = cw6 + cw3;
    static_assert(result == 9);

    cuda::std::same_as<cuda::std::__constant_wrapper<3>> decltype(auto) result2 = cw6 - cw3;
    static_assert(result2 == 3);

    cuda::std::same_as<cuda::std::__constant_wrapper<18>> decltype(auto) result3 = cw6 * cw3;
    static_assert(result3 == 18);

    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result4 = cw6 / cw3;
    static_assert(result4 == 2);

    cuda::std::same_as<cuda::std::__constant_wrapper<0>> decltype(auto) result5 = cw6 % cw3;
    static_assert(result5 == 0);

    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result6 = cw6 & cw3;
    static_assert(result6 == 2);

    cuda::std::same_as<cuda::std::__constant_wrapper<7>> decltype(auto) result7 = cw6 | cw3;
    static_assert(result7 == 7);

    cuda::std::same_as<cuda::std::__constant_wrapper<5>> decltype(auto) result8 = cw6 ^ cw3;
    static_assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    cuda::std::same_as<cuda::std::__constant_wrapper<48>> decltype(auto) result9 = cw6 << cw3;
    static_assert(result9 == 48);

    cuda::std::same_as<cuda::std::__constant_wrapper<0>> decltype(auto) result10 = cw6 >> cw3;
    static_assert(result10 == 0);

    // logical operations: int convertible to bool, so constant_wrapper overload is disabled
    // They are implicitly converted to bool and use built-in operators, resulting in a bool
    cuda::std::same_as<bool> decltype(auto) result11 = cw6 && cw3;
    assert(result11 == true);

    cuda::std::__constant_wrapper<0> cw0;
    cuda::std::same_as<bool> decltype(auto) result12 = cw0 || cw3;
    assert(result12 == true);
  }

  {
    // WithOps operations
    cuda::std::__constant_wrapper<WithOps{6}> cwWithOps6;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{9}>> decltype(auto) result =
      cwWithOps6 + cwWithOps3;
    static_assert(result.value.value == 9);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{3}>> decltype(auto) result2 =
      cwWithOps6 - cwWithOps3;
    static_assert(result2.value.value == 3);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{18}>> decltype(auto) result3 =
      cwWithOps6 * cwWithOps3;
    static_assert(result3.value.value == 18);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result4 =
      cwWithOps6 / cwWithOps3;
    static_assert(result4.value.value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{0}>> decltype(auto) result5 =
      cwWithOps6 % cwWithOps3;
    static_assert(result5.value.value == 0);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result6 =
      cwWithOps6 & cwWithOps3;
    static_assert(result6.value.value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{7}>> decltype(auto) result7 =
      cwWithOps6 | cwWithOps3;
    static_assert(result7.value.value == 7);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{5}>> decltype(auto) result8 =
      cwWithOps6 ^ cwWithOps3;
    static_assert(result8.value.value == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{48}>> decltype(auto) result9 =
      cwWithOps6 << cwWithOps3;
    static_assert(result9.value.value == 48);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{0}>> decltype(auto) result10 =
      cwWithOps6 >> cwWithOps3;
    static_assert(result10.value.value == 0);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result11 =
      cwWithOps6 && cwWithOps3;
    static_assert(result11.value.value == 1);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result12 =
      cwWithOps6 || cwWithOps3;
    static_assert(result12.value.value == 1);
  }

  {
    // Non-structural return types use implicit conversion
    cuda::std::__constant_wrapper<OpsReturnNonStructural{6}> cwOpt6;
    cuda::std::__constant_wrapper<OpsReturnNonStructural{3}> cwOpt3;

    cuda::std::same_as<NonStructural> decltype(auto) result = cwOpt6 + cwOpt3;
    assert(result.get() == 9);

    cuda::std::same_as<NonStructural> decltype(auto) result2 = cwOpt6 - cwOpt3;
    assert(result2.get() == 3);

    cuda::std::same_as<NonStructural> decltype(auto) result3 = cwOpt6 * cwOpt3;
    assert(result3.get() == 18);

    cuda::std::same_as<NonStructural> decltype(auto) result4 = cwOpt6 / cwOpt3;
    assert(result4.get() == 2);

    cuda::std::same_as<NonStructural> decltype(auto) result5 = cwOpt6 % cwOpt3;
    assert(result5.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) result6 = cwOpt6 & cwOpt3;
    assert(result6.get() == 2);

    cuda::std::same_as<NonStructural> decltype(auto) result7 = cwOpt6 | cwOpt3;
    assert(result7.get() == 7);

    cuda::std::same_as<NonStructural> decltype(auto) result8 = cwOpt6 ^ cwOpt3;
    assert(result8.get() == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    cuda::std::same_as<NonStructural> decltype(auto) result9 = cwOpt6 << cwOpt3;
    assert(result9.get() == 48);

    cuda::std::same_as<NonStructural> decltype(auto) result10 = cwOpt6 >> cwOpt3;
    assert(result10.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) result11 = cwOpt6 && cwOpt3;
    assert(result11.get() == 1);

    cuda::std::same_as<NonStructural> decltype(auto) result12 = cwOpt6 || cwOpt3;
    assert(result12.get() == 1);
  }

  {
    // Mix with runtime param: these operators are not used
    cuda::std::__constant_wrapper<6> cw6;
    int i = 3;

    cuda::std::same_as<int> decltype(auto) result = cw6 + i;
    assert(result == 9);

    cuda::std::same_as<int> decltype(auto) result2 = cw6 - i;
    assert(result2 == 3);

    cuda::std::same_as<int> decltype(auto) result3 = cw6 * i;
    assert(result3 == 18);

    cuda::std::same_as<int> decltype(auto) result4 = cw6 / i;
    assert(result4 == 2);

    cuda::std::same_as<int> decltype(auto) result5 = cw6 % i;
    assert(result5 == 0);

    cuda::std::same_as<int> decltype(auto) result6 = cw6 & i;
    assert(result6 == 2);

    cuda::std::same_as<int> decltype(auto) result7 = cw6 | i;
    assert(result7 == 7);

    cuda::std::same_as<int> decltype(auto) result8 = cw6 ^ i;
    assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    cuda::std::same_as<int> decltype(auto) result9 = cw6 << i;
    assert(result9 == 48);

    cuda::std::same_as<int> decltype(auto) result10 = cw6 >> i;
    assert(result10 == 0);

    cuda::std::same_as<bool> decltype(auto) result11 = cw6 && i;
    assert(result11 == true);

    cuda::std::__constant_wrapper<0> cw0;
    cuda::std::same_as<bool> decltype(auto) result12 = cw0 || i;
    assert(result12 == true);
  }

  {
    // with integral_constant
    cuda::std::__constant_wrapper<6> cw6;
    cuda::std::integral_constant<int, 3> ic3;

    cuda::std::same_as<cuda::std::__constant_wrapper<9>> decltype(auto) result = cw6 + ic3;
    static_assert(result == 9);

    cuda::std::same_as<cuda::std::__constant_wrapper<3>> decltype(auto) result2 = cw6 - ic3;
    static_assert(result2 == 3);

    cuda::std::same_as<cuda::std::__constant_wrapper<18>> decltype(auto) result3 = cw6 * ic3;
    static_assert(result3 == 18);

    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result4 = cw6 / ic3;
    static_assert(result4 == 2);

    cuda::std::same_as<cuda::std::__constant_wrapper<0>> decltype(auto) result5 = cw6 % ic3;
    static_assert(result5 == 0);

    cuda::std::same_as<cuda::std::__constant_wrapper<2>> decltype(auto) result6 = cw6 & ic3;
    static_assert(result6 == 2);

    cuda::std::same_as<cuda::std::__constant_wrapper<7>> decltype(auto) result7 = cw6 | ic3;
    static_assert(result7 == 7);

    cuda::std::same_as<cuda::std::__constant_wrapper<5>> decltype(auto) result8 = cw6 ^ ic3;
    static_assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    cuda::std::same_as<cuda::std::__constant_wrapper<48>> decltype(auto) result9 = cw6 << ic3;
    static_assert(result9 == 48);

    cuda::std::same_as<cuda::std::__constant_wrapper<0>> decltype(auto) result10 = cw6 >> ic3;
    static_assert(result10 == 0);

    // logical operations: int convertible to bool, so constant_wrapper overload is disabled
    // They are implicitly converted to bool and use built-in operators, resulting in a bool
    cuda::std::same_as<bool> decltype(auto) result11 = cw6 && ic3;
    assert(result11 == true);

    cuda::std::__constant_wrapper<0> cw0;
    cuda::std::same_as<bool> decltype(auto) result12 = cw0 || ic3;
    assert(result12 == true);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
