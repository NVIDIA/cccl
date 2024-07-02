//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// concept equality_comparable = // see below

#include <cuda/std/array>
#include <cuda/std/concepts>

#include "compare_types.h"

using cuda::std::equality_comparable;

namespace fundamentals
{
static_assert(equality_comparable<int>, "");
static_assert(equality_comparable<double>, "");
static_assert(equality_comparable<void*>, "");
static_assert(equality_comparable<char*>, "");
static_assert(equality_comparable<char const*>, "");
static_assert(equality_comparable<char volatile*>, "");
static_assert(equality_comparable<char const volatile*>, "");
static_assert(equality_comparable<wchar_t&>, "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(equality_comparable<char8_t const&>, "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(equality_comparable<char16_t volatile&>, "");
static_assert(equality_comparable<char32_t const volatile&>, "");
static_assert(equality_comparable<unsigned char&&>, "");
static_assert(equality_comparable<unsigned short const&&>, "");
static_assert(equality_comparable<unsigned int volatile&&>, "");
static_assert(equality_comparable<unsigned long const volatile&&>, "");
static_assert(equality_comparable<int[5]>, "");
static_assert(equality_comparable<int (*)(int)>, "");
static_assert(equality_comparable<int (&)(int)>, "");
static_assert(equality_comparable<int (*)(int) noexcept>, "");
static_assert(equality_comparable<int (&)(int) noexcept>, "");
static_assert(equality_comparable<cuda::std::nullptr_t>, "");

struct S
{};
static_assert(equality_comparable<int S::*>, "");
static_assert(equality_comparable<int (S::*)()>, "");
static_assert(equality_comparable<int (S::*)() noexcept>, "");
static_assert(equality_comparable<int (S::*)() &>, "");
static_assert(equality_comparable<int (S::*)() & noexcept>, "");
static_assert(equality_comparable<int (S::*)() &&>, "");
static_assert(equality_comparable < int(S::*)() && noexcept >, "");
static_assert(equality_comparable<int (S::*)() const>, "");
static_assert(equality_comparable<int (S::*)() const noexcept>, "");
static_assert(equality_comparable<int (S::*)() const&>, "");
static_assert(equality_comparable<int (S::*)() const & noexcept>, "");
static_assert(equality_comparable<int (S::*)() const&&>, "");
static_assert(equality_comparable < int(S::*)() const&& noexcept >, "");
static_assert(equality_comparable<int (S::*)() volatile>, "");
static_assert(equality_comparable<int (S::*)() volatile noexcept>, "");
static_assert(equality_comparable<int (S::*)() volatile&>, "");
static_assert(equality_comparable<int (S::*)() volatile & noexcept>, "");
static_assert(equality_comparable<int (S::*)() volatile&&>, "");
static_assert(equality_comparable < int(S::*)() volatile && noexcept >, "");
static_assert(equality_comparable<int (S::*)() const volatile>, "");
static_assert(equality_comparable<int (S::*)() const volatile noexcept>, "");
static_assert(equality_comparable<int (S::*)() const volatile&>, "");
static_assert(equality_comparable<int (S::*)() const volatile & noexcept>, "");
static_assert(equality_comparable<int (S::*)() const volatile&&>, "");
static_assert(equality_comparable < int(S::*)() const volatile&& noexcept >, "");

static_assert(!equality_comparable<void>, "");
} // namespace fundamentals

namespace standard_types
{
static_assert(equality_comparable<cuda::std::array<int, 10>>, "");
} // namespace standard_types

namespace types_fit_for_purpose
{
#if TEST_STD_VER > 2017
static_assert(equality_comparable<cxx20_member_eq>, "");
static_assert(equality_comparable<cxx20_friend_eq>, "");
#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(equality_comparable<member_three_way_comparable>, "");
#    ifndef __NVCC__ // nvbug3908399
static_assert(equality_comparable<friend_three_way_comparable>, "");
#    endif // !__NVCC_
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(equality_comparable<explicit_operators>, "");
static_assert(equality_comparable<different_return_types>, "");
static_assert(equality_comparable<one_member_one_friend>, "");
static_assert(equality_comparable<equality_comparable_with_ec1>, "");
#endif // TEST_STD_VER > 2017

static_assert(!equality_comparable<no_eq>, "");
static_assert(!equality_comparable<no_neq>, "");
static_assert(equality_comparable<no_lt>, "");
static_assert(equality_comparable<no_gt>, "");
static_assert(equality_comparable<no_le>, "");
static_assert(equality_comparable<no_ge>, "");

static_assert(!equality_comparable<wrong_return_type_eq>, "");
static_assert(!equality_comparable<wrong_return_type_ne>, "");
static_assert(equality_comparable<wrong_return_type_lt>, "");
static_assert(equality_comparable<wrong_return_type_gt>, "");
static_assert(equality_comparable<wrong_return_type_le>, "");
static_assert(equality_comparable<wrong_return_type_ge>, "");
static_assert(!equality_comparable<wrong_return_type>, "");

#if TEST_STD_VER > 2017
static_assert(!equality_comparable<cxx20_member_eq_operator_with_deleted_ne>, "");
static_assert(!equality_comparable<cxx20_friend_eq_operator_with_deleted_ne>, "");
#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(!equality_comparable<member_three_way_comparable_with_deleted_eq>, "");
static_assert(!equality_comparable<member_three_way_comparable_with_deleted_ne>, "");
static_assert(!equality_comparable<friend_three_way_comparable_with_deleted_eq>, "");
#    ifndef __NVCC__ // nvbug3908399
static_assert(!equality_comparable<friend_three_way_comparable_with_deleted_ne>, "");
#    endif // !__NVCC__

static_assert(!equality_comparable<eq_returns_explicit_bool>, "");
static_assert(!equality_comparable<ne_returns_explicit_bool>, "");
static_assert(equality_comparable<lt_returns_explicit_bool>, "");
static_assert(equality_comparable<gt_returns_explicit_bool>, "");
static_assert(equality_comparable<le_returns_explicit_bool>, "");
static_assert(equality_comparable<ge_returns_explicit_bool>, "");
static_assert(equality_comparable<returns_true_type>, "");
static_assert(equality_comparable<returns_int_ptr>, "");
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2017
} // namespace types_fit_for_purpose

int main(int, char**)
{
  return 0;
}
