//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class LHS, class RHS>
// concept assignable_from =
//   std::is_lvalue_reference_v<LHS> &&
//   std::common_reference_with<
//     const std::remove_reference_t<LHS>&,
//     const std::remove_reference_t<RHS>&> &&
//   requires (LHS lhs, RHS&& rhs) {
//     { lhs = std::forward<RHS>(rhs) } -> std::same_as<LHS>;
//   };

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "MoveOnly.h"
#include "test_macros.h"

struct NoCommonRef
{
  __host__ __device__ NoCommonRef& operator=(const int&);
};
static_assert(cuda::std::is_assignable_v<NoCommonRef&, const int&>, "");
static_assert(!cuda::std::assignable_from<NoCommonRef&, const int&>, ""); // no common reference type

struct Base
{};
struct Derived : Base
{};
static_assert(!cuda::std::assignable_from<Base*, Derived*>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*&&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const&&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*&&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const&&>, "");

struct VoidResultType
{
  __host__ __device__ void operator=(const VoidResultType&);
};
static_assert(cuda::std::is_assignable_v<VoidResultType&, const VoidResultType&>, "");
static_assert(!cuda::std::assignable_from<VoidResultType&, const VoidResultType&>, "");

struct ValueResultType
{
  __host__ __device__ ValueResultType operator=(const ValueResultType&);
};
static_assert(cuda::std::is_assignable_v<ValueResultType&, const ValueResultType&>, "");
static_assert(!cuda::std::assignable_from<ValueResultType&, const ValueResultType&>, "");

struct Locale
{
  __host__ __device__ const Locale& operator=(const Locale&);
};
static_assert(cuda::std::is_assignable_v<Locale&, const Locale&>, "");
static_assert(!cuda::std::assignable_from<Locale&, const Locale&>, "");

#ifdef TEST_COMPILER_MSVC_2017
#  pragma warning(disable : 4522) // multiple assignment operators defined
#endif // TEST_COMPILER_MSVC_2017
struct Tuple
{
  __host__ __device__ Tuple& operator=(const Tuple&);
  __host__ __device__ const Tuple& operator=(const Tuple&) const;
};
static_assert(!cuda::std::assignable_from<Tuple, const Tuple&>, "");
static_assert(cuda::std::assignable_from<Tuple&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<Tuple&&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<const Tuple, const Tuple&>, "");
static_assert(cuda::std::assignable_from<const Tuple&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<const Tuple&&, const Tuple&>, "");

// Finally, check a few simple cases.
static_assert(cuda::std::assignable_from<int&, int>, "");
static_assert(cuda::std::assignable_from<int&, int&>, "");
static_assert(cuda::std::assignable_from<int&, int&&>, "");
static_assert(!cuda::std::assignable_from<const int&, int>, "");
static_assert(!cuda::std::assignable_from<const int&, int&>, "");
static_assert(!cuda::std::assignable_from<const int&, int&&>, "");
static_assert(cuda::std::assignable_from<volatile int&, int>, "");
static_assert(cuda::std::assignable_from<volatile int&, int&>, "");
static_assert(cuda::std::assignable_from<volatile int&, int&&>, "");
static_assert(!cuda::std::assignable_from<int (&)[10], int>, "");
static_assert(!cuda::std::assignable_from<int (&)[10], int (&)[10]>, "");
#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::assignable_from<MoveOnly&, MoveOnly>, "");
#endif // !TEST_COMPILER_MSVC_2017
static_assert(!cuda::std::assignable_from<MoveOnly&, MoveOnly&>, "");
#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::assignable_from<MoveOnly&, MoveOnly&&>, "");
#endif // !TEST_COMPILER_MSVC_2017
static_assert(!cuda::std::assignable_from<void, int>, "");
static_assert(!cuda::std::assignable_from<void, void>, "");

int main(int, char**)
{
  return 0;
}
