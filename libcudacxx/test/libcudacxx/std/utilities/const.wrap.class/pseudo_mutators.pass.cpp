//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode

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

template <class T, class = void>
inline constexpr bool HasPreIncrement = false;
template <class T>
inline constexpr bool HasPreIncrement<T, cuda::std::void_t<decltype(++cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasPostIncrement = false;
template <class T>
inline constexpr bool HasPostIncrement<T, cuda::std::void_t<decltype(cuda::std::declval<T&>()++)>> = true;

template <class T, class = void>
inline constexpr bool HasPreDecrement = false;
template <class T>
inline constexpr bool HasPreDecrement<T, cuda::std::void_t<decltype(--cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasPostDecrement = false;
template <class T>
inline constexpr bool HasPostDecrement<T, cuda::std::void_t<decltype(cuda::std::declval<T&>()--)>> = true;

template <class L, class R, class = void>
inline constexpr bool HasPlusAssign = false;
template <class L, class R>
inline constexpr bool
  HasPlusAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() += cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasMinusAssign = false;
template <class L, class R>
inline constexpr bool
  HasMinusAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() -= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasMultiplyAssign = false;
template <class L, class R>
inline constexpr bool
  HasMultiplyAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() *= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasDivideAssign = false;
template <class L, class R>
inline constexpr bool
  HasDivideAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() /= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasModuloAssign = false;
template <class L, class R>
inline constexpr bool
  HasModuloAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() %= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasBitAndAssign = false;
template <class L, class R>
inline constexpr bool
  HasBitAndAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() &= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasBitOrAssign = false;
template <class L, class R>
inline constexpr bool
  HasBitOrAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() |= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasBitXorAssign = false;
template <class L, class R>
inline constexpr bool
  HasBitXorAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() ^= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasShiftLeftAssign = false;
template <class L, class R>
inline constexpr bool
  HasShiftLeftAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() <<= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasShiftRightAssign = false;
template <class L, class R>
inline constexpr bool
  HasShiftRightAssign<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() >>= cuda::std::declval<R&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptPreIncrement = false;
template <class T>
inline constexpr bool HasNoexceptPreIncrement<T, cuda::std::enable_if_t<noexcept(++cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptPostIncrement = false;
template <class T>
inline constexpr bool HasNoexceptPostIncrement<T, cuda::std::enable_if_t<noexcept(cuda::std::declval<T&>()++)>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptPreDecrement = false;
template <class T>
inline constexpr bool HasNoexceptPreDecrement<T, cuda::std::enable_if_t<noexcept(--cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptPostDecrement = false;
template <class T>
inline constexpr bool HasNoexceptPostDecrement<T, cuda::std::enable_if_t<noexcept(cuda::std::declval<T&>()--)>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptPlusAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptPlusAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() += cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptMinusAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptMinusAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() -= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptMultiplyAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptMultiplyAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() *= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptDivideAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptDivideAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() /= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptModuloAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptModuloAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() %= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptBitAndAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptBitAndAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() &= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptBitOrAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptBitOrAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() |= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptBitXorAssign = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptBitXorAssign<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() ^= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptShiftLeftAssign = false;
template <class L, class R>
inline constexpr bool HasNoexceptShiftLeftAssign<
  L,
  R,
  cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() <<= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptShiftRightAssign = false;
template <class L, class R>
inline constexpr bool HasNoexceptShiftRightAssign<
  L,
  R,
  cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() >>= cuda::std::declval<R&>())>> = true;

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

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

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
template <class T>
TEST_FUNC constexpr void lwg4383_f(T t)
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
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

TEST_FUNC constexpr bool test()
{
#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // WithOps increment/decrement
    cuda::std::__constant_wrapper<WithOps{5}> cwWithOps5;
    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{6}>> decltype(auto) result1 =
      ++cwWithOps5;
    static_assert(result1.__get().value == 6);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{6}>> decltype(auto) result2 =
      cwWithOps5++;
    static_assert(result2.__get().value == 6);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{4}>> decltype(auto) result3 =
      --cwWithOps5;
    static_assert(result3.__get().value == 4);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{4}>> decltype(auto) result4 =
      cwWithOps5--;
    static_assert(result4.__get().value == 4);
  }

// nvcc < 13.1 produces invalid source file for the host compilers. It replaces contexpr variables with their values
// which doesn't work for op=.
#  if !(_CCCL_CUDA_COMPILER(NVCC, <, 13, 1) && _CCCL_HOST_COMPILATION())
  {
    // WithOps compound assignments
    cuda::std::__constant_wrapper<WithOps{10}> cwWithOps10;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{13}>> decltype(auto) result1 =
      cwWithOps10 += cwWithOps3;
    static_assert(result1.__get().value == 13);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{7}>> decltype(auto) result2 =
      cwWithOps10 -= cwWithOps3;
    static_assert(result2.__get().value == 7);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{30}>> decltype(auto) result3 =
      cwWithOps10 *= cwWithOps3;
    static_assert(result3.__get().value == 30);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{3}>> decltype(auto) result4 =
      cwWithOps10 /= cwWithOps3;
    static_assert(result4.__get().value == 3);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result5 =
      cwWithOps10 %= cwWithOps3;
    static_assert(result5.__get().value == 1);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result6 =
      cwWithOps10 &= cwWithOps3;
    static_assert(result6.__get().value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{11}>> decltype(auto) result7 =
      cwWithOps10 |= cwWithOps3;
    static_assert(result7.__get().value == 11);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{9}>> decltype(auto) result8 =
      cwWithOps10 ^= cwWithOps3;
    static_assert(result8.__get().value == 9);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{80}>> decltype(auto) result9 =
      cwWithOps10 <<= cwWithOps3;
    static_assert(result9.__get().value == 80);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result10 =
      cwWithOps10 >>= cwWithOps3;
    static_assert(result10.__get().value == 1);
  }

  {
    // integral_constant compound assignments
    cuda::std::__constant_wrapper<WithOps{10}> cwWithOps10;
    cuda::std::integral_constant<WithOps, WithOps{3}> icWithOps3;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{13}>> decltype(auto) result1 =
      cwWithOps10 += icWithOps3;
    static_assert(result1.__get().value == 13);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{7}>> decltype(auto) result2 =
      cwWithOps10 -= icWithOps3;
    static_assert(result2.__get().value == 7);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{30}>> decltype(auto) result3 =
      cwWithOps10 *= icWithOps3;
    static_assert(result3.__get().value == 30);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{3}>> decltype(auto) result4 =
      cwWithOps10 /= icWithOps3;
    static_assert(result4.__get().value == 3);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result5 =
      cwWithOps10 %= icWithOps3;
    static_assert(result5.__get().value == 1);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{2}>> decltype(auto) result6 =
      cwWithOps10 &= icWithOps3;
    static_assert(result6.__get().value == 2);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{11}>> decltype(auto) result7 =
      cwWithOps10 |= icWithOps3;
    static_assert(result7.__get().value == 11);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{9}>> decltype(auto) result8 =
      cwWithOps10 ^= icWithOps3;
    static_assert(result8.__get().value == 9);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{80}>> decltype(auto) result9 =
      cwWithOps10 <<= icWithOps3;
    static_assert(result9.__get().value == 80);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{1}>> decltype(auto) result10 =
      cwWithOps10 >>= icWithOps3;
    static_assert(result10.__get().value == 1);
  }
#  endif // !(_CCCL_CUDA_COMPILER(NVCC, <, 13, 1) && _CCCL_HOST_COMPILATION())

  lwg4383();

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
