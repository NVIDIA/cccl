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
// concept totally_ordered_with;

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wc++17-extensions"
#  pragma clang diagnostic ignored "-Wordered-compare-function-pointers"
#endif

#include <cuda/std/array>
#include <cuda/std/concepts>

#include "compare_types.h"
#include "test_macros.h"

using cuda::std::equality_comparable_with;
using cuda::std::nullptr_t;
using cuda::std::totally_ordered_with;

template <class T, class U>
__host__ __device__ constexpr bool check_totally_ordered_with() noexcept
{
  constexpr bool result = totally_ordered_with<T, U>;
  static_assert(totally_ordered_with<U, T> == result, "");
  static_assert(totally_ordered_with<T, U const> == result, "");
  static_assert(totally_ordered_with<T const, U const> == result, "");
  static_assert(totally_ordered_with<T, U const&> == result, "");
  static_assert(totally_ordered_with<T const, U const&> == result, "");
  static_assert(totally_ordered_with<T&, U const> == result, "");
  static_assert(totally_ordered_with<T const&, U const> == result, "");
  static_assert(totally_ordered_with<T&, U const&> == result, "");
  static_assert(totally_ordered_with<T const&, U const&> == result, "");
  static_assert(totally_ordered_with<T, U const&&> == result, "");
  static_assert(totally_ordered_with<T const, U const&&> == result, "");
  static_assert(totally_ordered_with<T&, U const&&> == result, "");
  static_assert(totally_ordered_with<T const&, U const&&> == result, "");
  static_assert(totally_ordered_with<T&&, U const> == result, "");
  static_assert(totally_ordered_with<T const&&, U const> == result, "");
  static_assert(totally_ordered_with<T&&, U const&> == result, "");
  static_assert(totally_ordered_with<T const&&, U const&> == result, "");
  static_assert(totally_ordered_with<T&&, U const&&> == result, "");
  static_assert(totally_ordered_with<T const&&, U const&&> == result, "");
  return result;
}

namespace fundamentals
{
static_assert(check_totally_ordered_with<int, int>(), "");
static_assert(check_totally_ordered_with<int, bool>(), "");
static_assert(check_totally_ordered_with<int, char>(), "");
static_assert(check_totally_ordered_with<int, wchar_t>(), "");
static_assert(check_totally_ordered_with<int, double>(), "");
static_assert(!check_totally_ordered_with<int, int*>(), "");
static_assert(!check_totally_ordered_with<int, int[5]>(), "");
static_assert(!check_totally_ordered_with<int, int (*)()>(), "");
static_assert(!check_totally_ordered_with<int, int (&)()>(), "");

struct S
{};
static_assert(!check_totally_ordered_with<int, int S::*>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int, int (S::*)() const volatile&& noexcept > (), "");

static_assert(check_totally_ordered_with<int*, int*>(), "");
static_assert(check_totally_ordered_with<int*, int[5]>(), "");
static_assert(!check_totally_ordered_with<int*, int (*)()>(), "");
static_assert(!check_totally_ordered_with<int*, int (&)()>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int*, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int*, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int*, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int*, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int*, int (S::*)() const volatile&& noexcept > (), "");

static_assert(check_totally_ordered_with<int[5], int[5]>(), "");
static_assert(!check_totally_ordered_with<int[5], int (*)()>(), "");
static_assert(!check_totally_ordered_with<int[5], int (&)()>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int[5], int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int[5], int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int[5], int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int[5], int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int[5], int (S::*)() const volatile&& noexcept > (), "");

static_assert(check_totally_ordered_with<int (*)(), int (*)()>(), "");
static_assert(check_totally_ordered_with<int (*)(), int (&)()>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int (*)(), int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int (*)(), int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int (*)(), int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (*)(), int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int (*)(), int (S::*)() const volatile&& noexcept > (), "");
#ifdef INVESTIGATE_COMPILER_BUG
static_assert(check_totally_ordered_with<int (&)(), int (&)()>(), "");
#endif // INVESTIGATE_COMPILER_BUG
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int (&)(), int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int (&)(), int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int (&)(), int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (&)(), int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int (&)(), int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)(), int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)(), int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)(), int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)(), int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)(), int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() noexcept, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const noexcept, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const volatile & noexcept>(),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const volatile noexcept>(),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() volatile & noexcept>(),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const volatile&>(), "");
static_assert(
  !check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile noexcept,
              int (S::*)() volatile&& noexcept > (),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() & noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() & noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() & noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() & noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() & noexcept, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const& noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const& noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const& noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const & noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const& noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() const volatile & noexcept>(),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile & noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile & noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile & noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile & noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile & noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(
  !check_totally_ordered_with<int (S::*)() const volatile & noexcept, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile & noexcept, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile& noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile & noexcept, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile& noexcept,
              int (S::*)() const&& noexcept > (),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile & noexcept, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile& noexcept,
              int (S::*)() volatile&& noexcept > (),
              "");
static_assert(!check_totally_ordered_with<int (S::*)() const volatile & noexcept, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile& noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() &&, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &&, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &&, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &&, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() &&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() &&, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() const&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() volatile&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() const volatile&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() && noexcept, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with<int (S::*)() const&&, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&&, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() const&&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&&, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with < int(S::*)() const&& noexcept, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&& noexcept, int (S::*)() volatile&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&& noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&& noexcept, int (S::*)() const volatile&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() const&& noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() volatile&&, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&&, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<int (S::*)() volatile&&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile&&, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!check_totally_ordered_with < int(S::*)() volatile && noexcept, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile && noexcept, int (S::*)() const volatile&& > (), "");
static_assert(!check_totally_ordered_with < int(S::*)() volatile && noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

static_assert(!check_totally_ordered_with<int (S::*)() const volatile&&, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&&,
              int (S::*)() const volatile&& noexcept > (),
              "");
static_assert(!check_totally_ordered_with < int(S::*)() const volatile&& noexcept,
              int (S::*)() const volatile&& noexcept > (),
              "");

#if !defined(TEST_COMPILER_GCC) && defined(INVESTIGATE_COMPILER_BUG)
static_assert(!check_totally_ordered_with<nullptr_t, int>(), "");

static_assert(!check_totally_ordered_with<nullptr_t, int*>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int[]>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int[5]>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (*)()>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (&)()>(), "");
#endif

static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)()>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() volatile>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const volatile>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const volatile noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() &>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() & noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const&>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const & noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() volatile&>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const volatile&>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const volatile & noexcept>(), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() &&>(), "");
static_assert(!check_totally_ordered_with < nullptr_t, int (S::*)() && noexcept > (), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const&&>(), "");
static_assert(!check_totally_ordered_with < nullptr_t, int (S::*)() const&& noexcept > (), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() volatile&&>(), "");
static_assert(!check_totally_ordered_with < nullptr_t, int (S::*)() volatile&& noexcept > (), "");
static_assert(!check_totally_ordered_with<nullptr_t, int (S::*)() const volatile&&>(), "");
static_assert(!check_totally_ordered_with < nullptr_t, int (S::*)() const volatile&& noexcept > (), "");

static_assert(!equality_comparable_with<void, int>, "");
static_assert(!equality_comparable_with<void, int*>, "");
static_assert(!equality_comparable_with<void, nullptr_t>, "");
static_assert(!equality_comparable_with<void, int[5]>, "");
static_assert(!equality_comparable_with<void, int (*)()>, "");
static_assert(!equality_comparable_with<void, int (&)()>, "");
static_assert(!equality_comparable_with<void, int S::*>, "");
static_assert(!equality_comparable_with<void, int (S::*)()>, "");
static_assert(!equality_comparable_with<void, int (S::*)() noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() volatile>, "");
static_assert(!equality_comparable_with<void, int (S::*)() volatile noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const volatile>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const volatile noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() &>, "");
static_assert(!equality_comparable_with<void, int (S::*)() & noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const&>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const & noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() volatile&>, "");
static_assert(!equality_comparable_with<void, int (S::*)() volatile & noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const volatile&>, "");
static_assert(!equality_comparable_with<void, int (S::*)() const volatile & noexcept>, "");
static_assert(!equality_comparable_with<void, int (S::*)() &&>, "");
static_assert(!equality_comparable_with < void, int (S::*)() && noexcept >, "");
static_assert(!equality_comparable_with<void, int (S::*)() const&&>, "");
static_assert(!equality_comparable_with < void, int (S::*)() const&& noexcept >, "");
static_assert(!equality_comparable_with<void, int (S::*)() volatile&&>, "");
static_assert(!equality_comparable_with < void, int (S::*)() volatile&& noexcept >, "");
static_assert(!equality_comparable_with<void, int (S::*)() const volatile&&>, "");
static_assert(!equality_comparable_with < void, int (S::*)() const volatile&& noexcept >, "");
} // namespace fundamentals

namespace standard_types
{
static_assert(check_totally_ordered_with<cuda::std::array<int, 10>, cuda::std::array<int, 10>>(), "");
static_assert(!check_totally_ordered_with<cuda::std::array<int, 10>, cuda::std::array<double, 10>>(), "");
} // namespace standard_types

namespace types_fit_for_purpose
{
#if TEST_STD_VER > 2017
static_assert(!check_totally_ordered_with<cxx20_member_eq, cxx20_member_eq>(), "");
static_assert(!check_totally_ordered_with<cxx20_friend_eq, cxx20_friend_eq>(), "");
static_assert(!check_totally_ordered_with<cxx20_member_eq, cxx20_friend_eq>(), "");

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(check_totally_ordered_with<member_three_way_comparable, member_three_way_comparable>(), "");
#    ifndef __NVCC__ // nvbug3908399
static_assert(check_totally_ordered_with<friend_three_way_comparable, friend_three_way_comparable>(), "");
static_assert(!check_totally_ordered_with<member_three_way_comparable, friend_three_way_comparable>(), "");
#    endif // !__NVCC__
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2017

static_assert(check_totally_ordered_with<explicit_operators, explicit_operators>(), "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC has a bug where it considers the conversion with C++17
                                                        // and below
static_assert(!check_totally_ordered_with<equality_comparable_with_ec1, equality_comparable_with_ec1>(), "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(check_totally_ordered_with<different_return_types, different_return_types>(), "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC has a bug where it considers the conversion with C++17
                                                        // and below
static_assert(!check_totally_ordered_with<explicit_operators, equality_comparable_with_ec1>(), "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(check_totally_ordered_with<explicit_operators, different_return_types>(), "");

#if TEST_STD_VER > 2017
static_assert(!check_totally_ordered_with<one_way_eq, one_way_eq>(), "");
static_assert(cuda::std::common_reference_with<one_way_eq const&, explicit_operators const&>
                && !check_totally_ordered_with<one_way_eq, explicit_operators>(),
              "");

static_assert(!check_totally_ordered_with<one_way_ne, one_way_ne>(), "");
static_assert(cuda::std::common_reference_with<one_way_ne const&, explicit_operators const&>
                && !check_totally_ordered_with<one_way_ne, explicit_operators>(),
              "");

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
#    ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(check_totally_ordered_with<totally_ordered_with_others, partial_ordering_totally_ordered_with>(), "");
static_assert(check_totally_ordered_with<totally_ordered_with_others, weak_ordering_totally_ordered_with>(), "");
static_assert(check_totally_ordered_with<totally_ordered_with_others, strong_ordering_totally_ordered_with>(), "");
#    endif

static_assert(!check_totally_ordered_with<totally_ordered_with_others, eq_returns_explicit_bool>(), "");
static_assert(!check_totally_ordered_with<totally_ordered_with_others, ne_returns_explicit_bool>(), "");
static_assert(equality_comparable_with<totally_ordered_with_others, lt_returns_explicit_bool>
                && !check_totally_ordered_with<totally_ordered_with_others, lt_returns_explicit_bool>(),
              "");
static_assert(equality_comparable_with<totally_ordered_with_others, gt_returns_explicit_bool>
                && !check_totally_ordered_with<totally_ordered_with_others, gt_returns_explicit_bool>(),
              "");
static_assert(equality_comparable_with<totally_ordered_with_others, le_returns_explicit_bool>
                && !check_totally_ordered_with<totally_ordered_with_others, le_returns_explicit_bool>(),
              "");
static_assert(equality_comparable_with<totally_ordered_with_others, ge_returns_explicit_bool>
                && !check_totally_ordered_with<totally_ordered_with_others, ge_returns_explicit_bool>(),
              "");
static_assert(check_totally_ordered_with<totally_ordered_with_others, returns_true_type>(), "");
static_assert(check_totally_ordered_with<totally_ordered_with_others, returns_int_ptr>(), "");

static_assert(cuda::std::totally_ordered<no_lt_not_totally_ordered_with>
                && equality_comparable_with<totally_ordered_with_others, no_lt_not_totally_ordered_with>
                && !check_totally_ordered_with<totally_ordered_with_others, no_lt_not_totally_ordered_with>(),
              "");
static_assert(cuda::std::totally_ordered<no_gt_not_totally_ordered_with>
                && equality_comparable_with<totally_ordered_with_others, no_gt_not_totally_ordered_with>
                && !check_totally_ordered_with<totally_ordered_with_others, no_gt_not_totally_ordered_with>(),
              "");
static_assert(cuda::std::totally_ordered<no_le_not_totally_ordered_with>
                && equality_comparable_with<totally_ordered_with_others, no_le_not_totally_ordered_with>
                && !check_totally_ordered_with<totally_ordered_with_others, no_le_not_totally_ordered_with>(),
              "");
static_assert(cuda::std::totally_ordered<no_ge_not_totally_ordered_with>
                && equality_comparable_with<totally_ordered_with_others, no_ge_not_totally_ordered_with>
                && !check_totally_ordered_with<totally_ordered_with_others, no_ge_not_totally_ordered_with>(),
              "");
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2017
} // namespace types_fit_for_purpose

int main(int, char**)
{
  return 0;
}
