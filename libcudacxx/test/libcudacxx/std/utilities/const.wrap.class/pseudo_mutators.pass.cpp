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

// nvcc < 13.0 fails to match the operator overloads.
// UNSUPPORTED: nvcc-12

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper pseudo-mutators

// template<constexpr-param T>
//   constexpr auto operator++(this T) noexcept
//     -> constant_wrapper<++(T::value)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator++(this T, int) noexcept
//     -> constant_wrapper<(T::value++)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator--(this T) noexcept
//     -> constant_wrapper<--(T::value)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator--(this T, int) noexcept
//     -> constant_wrapper<(T::value--)> { return {}; }

// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator+=(this T, R) noexcept
//     -> constant_wrapper<(T::value += R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator-=(this T, R) noexcept
//     -> constant_wrapper<(T::value -= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator*=(this T, R) noexcept
//     -> constant_wrapper<(T::value *= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator/=(this T, R) noexcept
//     -> constant_wrapper<(T::value /= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator%=(this T, R) noexcept
//     -> constant_wrapper<(T::value %= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator&=(this T, R) noexcept
//     -> constant_wrapper<(T::value &= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator|=(this T, R) noexcept
//     -> constant_wrapper<(T::value |= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator^=(this T, R) noexcept
//     -> constant_wrapper<(T::value ^= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator<<=(this T, R) noexcept
//     -> constant_wrapper<(T::value <<= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator>>=(this T, R) noexcept
//     -> constant_wrapper<(T::value >>= R::value)> { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC constexpr auto operator++() const
  {
    return WithOps{value + 1};
  }
  TEST_FUNC constexpr auto operator++(int) const
  {
    return WithOps{value + 1};
  }
  TEST_FUNC constexpr auto operator--() const
  {
    return WithOps{value - 1};
  }
  TEST_FUNC constexpr auto operator--(int) const
  {
    return WithOps{value - 1};
  }

  TEST_FUNC constexpr auto operator+=(WithOps r) const
  {
    return WithOps{value + r.value};
  }
  TEST_FUNC constexpr auto operator-=(WithOps r) const
  {
    return WithOps{value - r.value};
  }
  TEST_FUNC constexpr auto operator*=(WithOps r) const
  {
    return WithOps{value * r.value};
  }
  TEST_FUNC constexpr auto operator/=(WithOps r) const
  {
    return WithOps{value / r.value};
  }
  TEST_FUNC constexpr auto operator%=(WithOps r) const
  {
    return WithOps{value % r.value};
  }
  TEST_FUNC constexpr auto operator&=(WithOps r) const
  {
    return WithOps{value & r.value};
  }
  TEST_FUNC constexpr auto operator|=(WithOps r) const
  {
    return WithOps{value | r.value};
  }
  TEST_FUNC constexpr auto operator^=(WithOps r) const
  {
    return WithOps{value ^ r.value};
  }
  TEST_FUNC constexpr auto operator<<=(WithOps r) const
  {
    return WithOps{value << r.value};
  }
  TEST_FUNC constexpr auto operator>>=(WithOps r) const
  {
    return WithOps{value >> r.value};
  }
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC constexpr auto operator++() const
  {
    return NonStructural{value + 1};
  }
  TEST_FUNC constexpr auto operator++(int) const
  {
    return NonStructural{value + 1};
  }
  TEST_FUNC constexpr auto operator--() const
  {
    return NonStructural{value - 1};
  }
  TEST_FUNC constexpr auto operator--(int) const
  {
    return NonStructural{value - 1};
  }

  TEST_FUNC constexpr auto operator+=(OpsReturnNonStructural r) const
  {
    return NonStructural{value + r.value};
  }
  TEST_FUNC constexpr auto operator-=(OpsReturnNonStructural r) const
  {
    return NonStructural{value - r.value};
  }
  TEST_FUNC constexpr auto operator*=(OpsReturnNonStructural r) const
  {
    return NonStructural{value * r.value};
  }
  TEST_FUNC constexpr auto operator/=(OpsReturnNonStructural r) const
  {
    return NonStructural{value / r.value};
  }
  TEST_FUNC constexpr auto operator%=(OpsReturnNonStructural r) const
  {
    return NonStructural{value % r.value};
  }
  TEST_FUNC constexpr auto operator&=(OpsReturnNonStructural r) const
  {
    return NonStructural{value & r.value};
  }
  TEST_FUNC constexpr auto operator|=(OpsReturnNonStructural r) const
  {
    return NonStructural{value | r.value};
  }
  TEST_FUNC constexpr auto operator^=(OpsReturnNonStructural r) const
  {
    return NonStructural{value ^ r.value};
  }
  TEST_FUNC constexpr auto operator<<=(OpsReturnNonStructural r) const
  {
    return NonStructural{value << r.value};
  }
  TEST_FUNC constexpr auto operator>>=(OpsReturnNonStructural r) const
  {
    return NonStructural{value >> r.value};
  }
};

struct NoOps
{};

template <class T>
concept HasPreIncrement = requires(T t) {
  { ++t };
};

template <class T>
concept HasPostIncrement = requires(T t) {
  { t++ };
};

template <class T>
concept HasPreDecrement = requires(T t) {
  { --t };
};

template <class T>
concept HasPostDecrement = requires(T t) {
  { t-- };
};

template <class L, class R>
concept HasPlusAssign = requires(L l, R r) {
  { l += r };
};

template <class L, class R>
concept HasMinusAssign = requires(L l, R r) {
  { l -= r };
};

template <class L, class R>
concept HasMultiplyAssign = requires(L l, R r) {
  { l *= r };
};

template <class L, class R>
concept HasDivideAssign = requires(L l, R r) {
  { l /= r };
};

template <class L, class R>
concept HasModuloAssign = requires(L l, R r) {
  { l %= r };
};

template <class L, class R>
concept HasBitAndAssign = requires(L l, R r) {
  { l &= r };
};

template <class L, class R>
concept HasBitOrAssign = requires(L l, R r) {
  { l |= r };
};

template <class L, class R>
concept HasBitXorAssign = requires(L l, R r) {
  { l ^= r };
};

template <class L, class R>
concept HasShiftLeftAssign = requires(L l, R r) {
  { l <<= r };
};

template <class L, class R>
concept HasShiftRightAssign = requires(L l, R r) {
  { l >>= r };
};

template <class T>
concept HasNoexceptPreIncrement = requires(T t) {
  { ++t } noexcept;
};

template <class T>
concept HasNoexceptPostIncrement = requires(T t) {
  { t++ } noexcept;
};

template <class T>
concept HasNoexceptPreDecrement = requires(T t) {
  { --t } noexcept;
};

template <class T>
concept HasNoexceptPostDecrement = requires(T t) {
  { t-- } noexcept;
};

template <class L, class R>
concept HasNoexceptPlusAssign = requires(L l, R r) {
  { l += r } noexcept;
};

template <class L, class R>
concept HasNoexceptMinusAssign = requires(L l, R r) {
  { l -= r } noexcept;
};

template <class L, class R>
concept HasNoexceptMultiplyAssign = requires(L l, R r) {
  { l *= r } noexcept;
};

template <class L, class R>
concept HasNoexceptDivideAssign = requires(L l, R r) {
  { l /= r } noexcept;
};

template <class L, class R>
concept HasNoexceptModuloAssign = requires(L l, R r) {
  { l %= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitAndAssign = requires(L l, R r) {
  { l &= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitOrAssign = requires(L l, R r) {
  { l |= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitXorAssign = requires(L l, R r) {
  { l ^= r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftLeftAssign = requires(L l, R r) {
  { l <<= r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftRightAssign = requires(L l, R r) {
  { l >>= r } noexcept;
};

// Pseudo-mutators does work with int as built-in types mutating operators are const
static_assert(!HasPreIncrement<cuda::std::__constant_wrapper<6>>);
static_assert(!HasPostIncrement<cuda::std::__constant_wrapper<6>>);
static_assert(!HasPreDecrement<cuda::std::__constant_wrapper<6>>);
static_assert(!HasPostDecrement<cuda::std::__constant_wrapper<6>>);

static_assert(!HasPlusAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasMinusAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasMultiplyAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasDivideAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasModuloAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasBitAndAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasBitOrAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasBitXorAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasShiftLeftAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);
static_assert(!HasShiftRightAssign<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<1>>);

// NoOps - pseudo-mutators shouldn't work without supporting operators
static_assert(!HasPreIncrement<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasPostIncrement<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasPreDecrement<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasPostDecrement<cuda::std::__constant_wrapper<NoOps{}>>);

static_assert(!HasPlusAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasMinusAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasMultiplyAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasDivideAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasModuloAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitAndAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitOrAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitXorAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasShiftLeftAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasShiftRightAssign<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);

// Pseudo-mutators work with WithOps types
static_assert(HasNoexceptPreIncrement<cuda::std::__constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPostIncrement<cuda::std::__constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPreDecrement<cuda::std::__constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPostDecrement<cuda::std::__constant_wrapper<WithOps{6}>>);

static_assert(
  HasNoexceptPlusAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptMinusAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptMultiplyAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptDivideAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptModuloAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptBitAndAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptBitOrAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptBitXorAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptShiftLeftAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{1}>>);
static_assert(
  HasNoexceptShiftRightAssign<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{1}>>);

// clang-format off
// Non-structural return types cannot use implicit conversions too because they are member functions and cannot be found through ADL
static_assert(!HasPreIncrement<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPostIncrement<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPreDecrement<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPostDecrement<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>>);

static_assert(!HasPlusAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasMinusAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasMultiplyAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasDivideAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasModuloAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitAndAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitOrAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitXorAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasShiftLeftAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasShiftRightAssign<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{1}>>);
// clang-format on

// LWG 4383. constant_wrapper's pseudo-mutators are underconstrained
// https://cplusplus.github.io/LWG/issue4383
TEST_FUNC constexpr void lwg4383_f(auto t)
{
  if constexpr (requires { +t; }) // ok
  {
    +t;
  }
  if constexpr (requires { -t; }) // ok
  {
    -t;
  }
  if constexpr (requires { ++t; }) // no hard error
  {
    ++t;
  }
  if constexpr (requires { --t; }) // no hard error
  {
    --t;
  }
}

struct S
{
  TEST_FUNC /* constexpr */ int operator+() const
  {
    return 0;
  }
  TEST_FUNC /* constexpr */ int operator++()
  {
    return 0;
  }
  TEST_FUNC constexpr void operator-() const {}
  TEST_FUNC constexpr void operator--() {}
};

TEST_FUNC constexpr void lwg4383()
{
  lwg4383_f(cuda::std::__cw<S{}>);
}

TEST_FUNC constexpr bool test()
{
  {
    // WithOps increment/decrement
    cuda::std::__constant_wrapper<WithOps{5}> cwWithOps5;
    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{6}>> decltype(auto) result1 =
      ++cwWithOps5;
    static_assert(result1.value.value == 6);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{6}>> decltype(auto) result2 =
      cwWithOps5++;
    static_assert(result2.value.value == 6);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{4}>> decltype(auto) result3 =
      --cwWithOps5;
    static_assert(result3.value.value == 4);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{4}>> decltype(auto) result4 =
      cwWithOps5--;
    static_assert(result4.value.value == 4);
  }

// nvcc == 13.0 produces invalid source file for the host compilers. It replaces contexpr variables with their values
// which doesn't work for op=.
#if !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0)
  {
    // WithOps compound assignments
    cuda::std::__constant_wrapper<WithOps{10}> cwWithOps10;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{13}>> decltype(auto) result1 =
      cwWithOps10 += cwWithOps3;
    static_assert(result1.value.value == 13);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{7}>> decltype(auto) result2 =
      cwWithOps10 -= cwWithOps3;
    static_assert(result2.value.value == 7);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{30}>> decltype(auto) result3 =
      cwWithOps10 *= cwWithOps3;
    static_assert(result3.value.value == 30);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{3}>> decltype(auto) result4 =
      cwWithOps10 /= cwWithOps3;
    static_assert(result4.value.value == 3);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result5 =
      cwWithOps10 %= cwWithOps3;
    static_assert(result5.value.value == 1);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result6 =
      cwWithOps10 &= cwWithOps3;
    static_assert(result6.value.value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{11}>> decltype(auto) result7 =
      cwWithOps10 |= cwWithOps3;
    static_assert(result7.value.value == 11);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{9}>> decltype(auto) result8 =
      cwWithOps10 ^= cwWithOps3;
    static_assert(result8.value.value == 9);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{80}>> decltype(auto) result9 =
      cwWithOps10 <<= cwWithOps3;
    static_assert(result9.value.value == 80);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result10 =
      cwWithOps10 >>= cwWithOps3;
    static_assert(result10.value.value == 1);
  }

  {
    // integral_constant compound assignments
    cuda::std::__constant_wrapper<WithOps{10}> cwWithOps10;
    cuda::std::integral_constant<WithOps, WithOps{3}> icWithOps3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{13}>> decltype(auto) result1 =
      cwWithOps10 += icWithOps3;
    static_assert(result1.value.value == 13);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{7}>> decltype(auto) result2 =
      cwWithOps10 -= icWithOps3;
    static_assert(result2.value.value == 7);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{30}>> decltype(auto) result3 =
      cwWithOps10 *= icWithOps3;
    static_assert(result3.value.value == 30);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{3}>> decltype(auto) result4 =
      cwWithOps10 /= icWithOps3;
    static_assert(result4.value.value == 3);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result5 =
      cwWithOps10 %= icWithOps3;
    static_assert(result5.value.value == 1);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result6 =
      cwWithOps10 &= icWithOps3;
    static_assert(result6.value.value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{11}>> decltype(auto) result7 =
      cwWithOps10 |= icWithOps3;
    static_assert(result7.value.value == 11);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{9}>> decltype(auto) result8 =
      cwWithOps10 ^= icWithOps3;
    static_assert(result8.value.value == 9);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{80}>> decltype(auto) result9 =
      cwWithOps10 <<= icWithOps3;
    static_assert(result9.value.value == 80);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result10 =
      cwWithOps10 >>= icWithOps3;
    static_assert(result10.value.value == 1);
  }
#endif // !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 0)

  lwg4383();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
