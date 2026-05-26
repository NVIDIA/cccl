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

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator,(L, R) noexcept = delete;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Suppress warnings regarding unused value in A, B expression.
TEST_NV_DIAG_SUPPRESS(174)
TEST_DIAG_SUPPRESS_CLANG("-Wunused-value")
TEST_DIAG_SUPPRESS_GCC("-Wunused-value")
TEST_DIAG_SUPPRESS_NVHPC(expr_has_no_effect)

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator,(const WithOps& /*l*/, WithOps r)
  {
    return WithOps{r.value};
  }
};

struct NoOps
{};

// cudafe++ has problems parsing this concept, so we need to use SFINAE instead. See nvbug 6183205.
// template <class L, class R>
// concept HasComma = requires(L l, R r) {
//   { l, r };
// };
template <class L, class R, class = void>
inline constexpr bool HasComma = false;
template <class L, class R>
inline constexpr bool HasComma<L, R, cuda::std::void_t<decltype(cuda::std::declval<L>(), cuda::std::declval<R>())>> =
  true;

// Comma operator is deleted for constant_wrapper operands
static_assert(!HasComma<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(!HasComma<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(!HasComma<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);

// Mixed operands - one constant_wrapper, one runtime type (uses built-in operator)
static_assert(HasComma<cuda::std::__constant_wrapper<42>, int>);
static_assert(HasComma<int, cuda::std::__constant_wrapper<42>>);

TEST_FUNC constexpr bool test()
{
  {
    // only mixed with runtime parameters
    cuda::std::__constant_wrapper<42> cw42;
    int i                                           = 0;
    cuda::std::same_as<int&> decltype(auto) result1 = (cw42, i);
    assert(result1 == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
