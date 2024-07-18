//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class In>
// concept cuda::std::weakly_incrementable;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "../incrementable.h"
#include "test_macros.h"

static_assert(cuda::std::weakly_incrementable<int>);
static_assert(cuda::std::weakly_incrementable<int*>);
static_assert(cuda::std::weakly_incrementable<int**>);
static_assert(!cuda::std::weakly_incrementable<int[]>);
static_assert(!cuda::std::weakly_incrementable<int[10]>);
static_assert(!cuda::std::weakly_incrementable<double>);
static_assert(!cuda::std::weakly_incrementable<int&>);
static_assert(!cuda::std::weakly_incrementable<int()>);
static_assert(!cuda::std::weakly_incrementable<int (*)()>);
static_assert(!cuda::std::weakly_incrementable<int (&)()>);
#ifndef TEST_COMPILER_GCC
static_assert(!cuda::std::weakly_incrementable<bool>);
#endif

struct S
{};
static_assert(!cuda::std::weakly_incrementable<int S::*>);

#define CHECK_POINTER_TO_MEMBER_FUNCTIONS(qualifier)                                  \
  static_assert(!cuda::std::weakly_incrementable<int (S::*)() qualifier>);            \
  static_assert(!cuda::std::weakly_incrementable<int (S::*)() qualifier noexcept>);   \
  static_assert(!cuda::std::weakly_incrementable<int (S::*)() qualifier&>);           \
  static_assert(!cuda::std::weakly_incrementable<int (S::*)() qualifier & noexcept>); \
  static_assert(!cuda::std::weakly_incrementable<int (S::*)() qualifier&&>);          \
  static_assert(!cuda::std::weakly_incrementable < int(S::*)() qualifier && noexcept >);

#define NO_QUALIFIER
CHECK_POINTER_TO_MEMBER_FUNCTIONS(NO_QUALIFIER);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(volatile);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const volatile);

static_assert(cuda::std::weakly_incrementable<postfix_increment_returns_void>);
static_assert(cuda::std::weakly_incrementable<postfix_increment_returns_copy>);
static_assert(cuda::std::weakly_incrementable<has_integral_minus>);
static_assert(cuda::std::weakly_incrementable<has_distinct_difference_type_and_minus>);
static_assert(!cuda::std::weakly_incrementable<missing_difference_type>);
static_assert(!cuda::std::weakly_incrementable<floating_difference_type>);
static_assert(!cuda::std::weakly_incrementable<non_const_minus>);
static_assert(!cuda::std::weakly_incrementable<non_integral_minus>);
static_assert(!cuda::std::weakly_incrementable<bad_difference_type_good_minus>);
static_assert(!cuda::std::weakly_incrementable<not_movable>);
static_assert(!cuda::std::weakly_incrementable<preinc_not_declared>);
static_assert(!cuda::std::weakly_incrementable<postinc_not_declared>);
static_assert(cuda::std::weakly_incrementable<not_default_initializable>);
static_assert(cuda::std::weakly_incrementable<incrementable_with_difference_type>);
static_assert(cuda::std::weakly_incrementable<incrementable_without_difference_type>);
static_assert(cuda::std::weakly_incrementable<difference_type_and_void_minus>);
#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::weakly_incrementable<noncopyable_with_difference_type>);
static_assert(cuda::std::weakly_incrementable<noncopyable_without_difference_type>);
static_assert(cuda::std::weakly_incrementable<noncopyable_with_difference_type_and_minus>);
#endif // TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  return 0;
}
