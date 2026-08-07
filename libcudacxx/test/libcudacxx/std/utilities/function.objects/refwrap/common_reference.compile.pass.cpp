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

#if !TEST_COMPILER(NVRTC)
#  include <functional>
#endif // !TEST_COMPILER(NVRTC)

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

using cuda::std::common_reference_t;
using cuda::std::is_same_v;

// derived-base and implicit convertibles
struct B
{};
struct D : B
{};
struct C
{
  TEST_FUNC operator B&() const;
};

// Check that RefW<T> behaves the same as T& in common_reference.
template <template <class> class RefW, class T>
TEST_FUNC constexpr bool test_same_as_reference()
{
  using R1 = common_reference_t<T&, T&>;
  using R2 = common_reference_t<T&, T const&>;
  using R3 = common_reference_t<T&, T&&>;
  using R4 = common_reference_t<T&, T const&&>;
  using R5 = common_reference_t<T&, T>;

  // clang-format off
  static_assert(is_same_v<R1, common_reference_t<RefW<T>, T&>>);
  static_assert(is_same_v<R2, common_reference_t<RefW<T>, T const&>>);
  static_assert(is_same_v<R3, common_reference_t<RefW<T>, T&&>>);
  static_assert(is_same_v<R4, common_reference_t<RefW<T>, T const&&>>);
  static_assert(is_same_v<R5, common_reference_t<RefW<T>, T>>);

  // commute:
  static_assert(is_same_v<R1, common_reference_t<T&,        RefW<T>>>);
  static_assert(is_same_v<R2, common_reference_t<T const&,  RefW<T>>>);
  static_assert(is_same_v<R3, common_reference_t<T&&,       RefW<T>>>);
  static_assert(is_same_v<R4, common_reference_t<T const&&, RefW<T>>>);
  static_assert(is_same_v<R5, common_reference_t<T,         RefW<T>>>);

  // reference qualification of reference_wrapper is irrelevant
  static_assert(is_same_v<R1, common_reference_t<RefW<T>&,        T&>>);
  static_assert(is_same_v<R1, common_reference_t<RefW<T>,         T&>>);
  static_assert(is_same_v<R1, common_reference_t<RefW<T> const&,  T&>>);
  static_assert(is_same_v<R1, common_reference_t<RefW<T>&&,       T&>>);
  static_assert(is_same_v<R1, common_reference_t<RefW<T> const&&, T&>>);
  // clang-format on

  return true;
}

template <template <class> class RefW>
TEST_FUNC constexpr bool test_common_reference()
{
  using Ri   = RefW<int>;
  using RRi  = RefW<RefW<int>>;
  using RRRi = RefW<RefW<RefW<int>>>;

  // clang-format off
  static_assert(check<int&,       RefW<int>,       int&>);
  static_assert(check<int const&, RefW<int>,       int const&>);
  static_assert(check<int const&, RefW<int const>, int&>);
  static_assert(check<int const&, RefW<int const>, int const&>);
  static_assert(check<int&,       RefW<int> const&, int&>);
  static_assert(check<const volatile int&, RefW<const volatile int>, const volatile int&>);

  static_assert(check<B&,       RefW<B>,       D&>);
  static_assert(check<B const&, RefW<B>,       D const&>);
  static_assert(check<B const&, RefW<B const>, D const&>);

  static_assert(check<B&,       RefW<D>,       B&>);
  static_assert(check<B const&, RefW<D>,       B const&>);
  static_assert(check<B const&, RefW<D const>, B const&>);

  static_assert( check<B&,       RefW<B>,       C&>);
// MSVC does not agree on these COND-RES results; the cause is the DevCom-1627396 workaround in
// <cuda/std/__type_traits/common_reference.h>, whose fix is tracked separately and is deliberately
// not part of this P2655R3 change.
#if !TEST_COMPILER(MSVC)
  static_assert( check<B&,       RefW<B>,       C>);
  static_assert( check<B const&, RefW<B const>, C>);
#endif // !TEST_COMPILER(MSVC)
  static_assert(!check<B&,       RefW<C>,       B&>); // RefW<C> cannot be converted to B&
  static_assert( check<B&,       RefW<B>,       C const&>); // was const B& before P2655R3

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

  // reference_wrapper as both args is unaffected: without the mutually exclusive
  // exists_with conditions this would be an ambiguous partial specialization.
  static_assert(check<RefW<int>&, RefW<int>&, RefW<int>&>);

  // double wrap is unaffected.
  static_assert(check<RefW<int>&, RefW<RefW<int>>, RefW<int>&>);
  // clang-format on

  return true;
}

static_assert(test_common_reference<cuda::std::reference_wrapper>());
static_assert(test_same_as_reference<cuda::std::reference_wrapper, int>());
static_assert(test_same_as_reference<cuda::std::reference_wrapper, cuda::std::reference_wrapper<int>>());

#if !TEST_COMPILER(NVRTC)
static_assert(test_common_reference<::std::reference_wrapper>());
static_assert(test_same_as_reference<::std::reference_wrapper, int>());
static_assert(test_same_as_reference<::std::reference_wrapper, ::std::reference_wrapper<int>>());
#endif // !TEST_COMPILER(NVRTC)

int main(int, char**)
{
  return 0;
}
