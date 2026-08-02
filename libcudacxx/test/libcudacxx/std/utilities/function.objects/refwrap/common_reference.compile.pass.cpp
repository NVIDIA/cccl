//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// P2655R3: common_reference specializations for reference_wrapper

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T1, class T2, class = void>
inline constexpr bool has_common_reference = false;

template <class T1, class T2>
inline constexpr bool has_common_reference<T1, T2, cuda::std::void_t<cuda::std::common_reference_t<T1, T2>>> = true;

// Deliberately SFINAE-friendly: `check` must yield false, not a hard error, when the two
// types have no common reference at all.
template <class Result, class T1, class T2, class = void>
inline constexpr bool check_one = false;

template <class Result, class T1, class T2>
inline constexpr bool check_one<Result, T1, T2, cuda::std::void_t<cuda::std::common_reference_t<T1, T2>>> =
  cuda::std::is_same_v<Result, cuda::std::common_reference_t<T1, T2>>;

template <class Result, class T1, class T2>
inline constexpr bool check = check_one<Result, T1, T2> && check_one<Result, T2, T1>;

template <class T1, class T2>
inline constexpr bool check_none = !has_common_reference<T1, T2> && !has_common_reference<T2, T1>;

// https://eel.is/c++draft/meta.trans#other-2.4
template <class X, class Y>
using CondRes = decltype(false ? cuda::std::declval<X (&)()>()() : cuda::std::declval<Y (&)()>()());

template <class T>
using Ref = cuda::std::reference_wrapper<T>;

using cuda::std::common_reference_t;
using cuda::std::is_same_v;

// clang-format off
static_assert(check<int&,       Ref<int>,       int&>);
static_assert(check<int const&, Ref<int>,       int const&>);
static_assert(check<int const&, Ref<int const>, int&>);
static_assert(check<int const&, Ref<int const>, int const&>);
static_assert(check<int&,       Ref<int> const&, int&>);
static_assert(check<const volatile int&, Ref<const volatile int>, const volatile int&>);

// derived-base and implicit convertibles
struct B
{};
struct D : B
{};
struct C
{
  TEST_FUNC operator B&() const;
};

static_assert(check<B&,       Ref<B>,       D&>);
static_assert(check<B const&, Ref<B>,       D const&>);
static_assert(check<B const&, Ref<B const>, D const&>);

static_assert(check<B&,       Ref<D>,       B&>);
static_assert(check<B const&, Ref<D>,       B const&>);
static_assert(check<B const&, Ref<D const>, B const&>);

static_assert(is_same_v<B&,       CondRes<Ref<D>,       B&>>);
static_assert(is_same_v<B const&, CondRes<Ref<D>,       B const&>>);
static_assert(is_same_v<B const&, CondRes<Ref<D const>, B const&>>);

static_assert( check<B&,       Ref<B>,       C&>);
static_assert( check<B&,       Ref<B>,       C>);
static_assert( check<B const&, Ref<B const>, C>);
static_assert(!check<B&,       Ref<C>,       B&>); // Ref<C> cannot be converted to B&
static_assert( check<B&,       Ref<B>,       C const&>); // was const B& before P2655R3

using Ri   = Ref<int>;
using RRi  = Ref<Ref<int>>;
using RRRi = Ref<Ref<Ref<int>>>;
static_assert(check<Ri&,  Ri&,  RRi>);
static_assert(check<RRi&, RRi&, RRRi>);
static_assert(check<Ri,   Ri,   RRi>);
static_assert(check<RRi,  RRi,  RRRi>);

static_assert(check_none<int&, RRi>);
static_assert(check_none<int,  RRi>);
static_assert(check_none<int&, RRRi>);
static_assert(check_none<int,  RRRi>);

static_assert(check_none<Ri&, RRRi>);
static_assert(check_none<Ri,  RRRi>);

template <typename T>
struct Test
{
  // Check that reference_wrapper<T> behaves the same as T& in common_reference.

  using R1 = common_reference_t<T&, T&>;
  using R2 = common_reference_t<T&, T const&>;
  using R3 = common_reference_t<T&, T&&>;
  using R4 = common_reference_t<T&, T const&&>;
  using R5 = common_reference_t<T&, T>;

  static_assert(is_same_v<R1, common_reference_t<Ref<T>, T&>>);
  static_assert(is_same_v<R2, common_reference_t<Ref<T>, T const&>>);
  static_assert(is_same_v<R3, common_reference_t<Ref<T>, T&&>>);
  static_assert(is_same_v<R4, common_reference_t<Ref<T>, T const&&>>);
  static_assert(is_same_v<R5, common_reference_t<Ref<T>, T>>);

  // commute:
  static_assert(is_same_v<R1, common_reference_t<T&,        Ref<T>>>);
  static_assert(is_same_v<R2, common_reference_t<T const&,  Ref<T>>>);
  static_assert(is_same_v<R3, common_reference_t<T&&,       Ref<T>>>);
  static_assert(is_same_v<R4, common_reference_t<T const&&, Ref<T>>>);
  static_assert(is_same_v<R5, common_reference_t<T,         Ref<T>>>);

  // reference qualification of reference_wrapper is irrelevant
  static_assert(is_same_v<R1, common_reference_t<Ref<T>&,        T&>>);
  static_assert(is_same_v<R1, common_reference_t<Ref<T>,         T&>>);
  static_assert(is_same_v<R1, common_reference_t<Ref<T> const&,  T&>>);
  static_assert(is_same_v<R1, common_reference_t<Ref<T>&&,       T&>>);
  static_assert(is_same_v<R1, common_reference_t<Ref<T> const&&, T&>>);
};
// clang-format on

// Instantiate above checks:
template struct Test<int>;
template struct Test<cuda::std::reference_wrapper<int>>;

// reference_wrapper as both args is unaffected: without the mutually exclusive
// exists_with conditions this would be an ambiguous partial specialization.
static_assert(check<Ref<int>&, Ref<int>&, Ref<int>&>);

// double wrap is unaffected.
static_assert(check<Ref<int>&, Ref<Ref<int>>, Ref<int>&>);

int main(int, char**)
{
  return 0;
}
