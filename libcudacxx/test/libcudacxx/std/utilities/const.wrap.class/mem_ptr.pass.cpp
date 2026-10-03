//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// nvcc + msvc + c++17 combination fails to compile with error:
//   template instantiation resulted in unexpected function type
// XFAIL: nvcc && msvc && c++17

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator->*(L, R) noexcept -> constant_wrapper<L::value->*(R::value)>
//     { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

// gcc warns about some unnecessary parentheses being emitted by cudefe++.
TEST_DIAG_SUPPRESS_GCC("-Wparentheses")

struct S
{
  int member = 42;
};

constexpr S s_value{};

template <class L, class R, class = void>
inline constexpr bool HasPtrToMem = false;
template <class L, class R>
inline constexpr bool
  HasPtrToMem<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>()->*cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptPtrToMem = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptPtrToMem<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>()->*cuda::std::declval<R&>())>> =
    true;

struct WithOps
{
  int value;
  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator->*(WithOps w, int WithOps::* pm)
  {
    return w.value + (&w)->*pm;
  }
};

struct OpsReturnNonStructural
{
  int value;
  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator->*(OpsReturnNonStructural o, int OpsReturnNonStructural::* pm)
  {
    return NonStructural{o.value + (&o)->*pm};
  }
};

struct NoOps
{};

static_assert(HasPtrToMem<cuda::std::constant_wrapper<&s_value>, cuda::std::constant_wrapper<&S::member>>);
static_assert(HasNoexceptPtrToMem<cuda::std::constant_wrapper<&s_value>, cuda::std::constant_wrapper<&S::member>>);

static_assert(HasPtrToMem<cuda::std::constant_wrapper<&s_value>, int S::*>);
static_assert(!HasPtrToMem<cuda::std::constant_wrapper<&s_value>, int>);

// Runtime data-member pointers must preserve all pointee cv-qualifiers.
template <class T>
using RuntimeMemberResult =
  decltype(cuda::std::declval<cuda::std::constant_wrapper<static_cast<T*>(nullptr)>>()->*cuda::std::declval<int S::*>());

static_assert(cuda::std::same_as<int&, RuntimeMemberResult<S>>);
static_assert(cuda::std::same_as<const int&, RuntimeMemberResult<const S>>);
static_assert(cuda::std::same_as<volatile int&, RuntimeMemberResult<volatile S>>);
static_assert(cuda::std::same_as<const volatile int&, RuntimeMemberResult<const volatile S>>);
static_assert(!HasPtrToMem<cuda::std::constant_wrapper<&s_value>, int NoOps::*>);

TEST_FUNC constexpr bool test()
{
  {
    // use builtin operator->*
    cuda::std::constant_wrapper<(&s_value)> cwS{};
    cuda::std::constant_wrapper<&S::member> cwPM{};
    decltype(auto) result1 = cwS->*cwPM;
    static_assert(cuda::std::same_as<cuda::std::constant_wrapper<42>, decltype(result1)>);
    static_assert(result1 == 42);
  }

  {
    // mix runtime and constant_wrapper parameters
    cuda::std::constant_wrapper<(&s_value)> cwS{};
    int S::* pm            = &S::member;
    decltype(auto) result1 = cwS->*pm;
    static_assert(cuda::std::same_as<const int&, decltype(result1)>);
    assert(result1 == 42);
  }

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
  {
    // custom operator->*
    cuda::std::constant_wrapper<WithOps{42}> cwWO;
    cuda::std::constant_wrapper<&WithOps::value> cwPM;
    cuda::std::same_as<cuda::std::constant_wrapper<84>> decltype(auto) result1 = cwWO->*cwPM;
    static_assert(result1 == 84);
  }

  {
    // Return non-structural type
    // Will use underlying type's runtime operators
    cuda::std::constant_wrapper<OpsReturnNonStructural{42}> cwORNS;
    cuda::std::constant_wrapper<&OpsReturnNonStructural::value> cwPM;
    cuda::std::same_as<NonStructural> decltype(auto) result1 = cwORNS->*cwPM;
    assert(result1.get() == 84);
  }
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // integral_constant
    cuda::std::constant_wrapper<(&s_value)> cwS{};
    cuda::std::integral_constant<int S::*, &S::member> icPM{};
    decltype(auto) result1 = cwS->*icPM;
    static_assert(cuda::std::same_as<cuda::std::constant_wrapper<42>, decltype(result1)>);
    static_assert(result1 == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
