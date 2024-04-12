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
// concept indirectly_readable;

#include "../incrementable.h"

#include <cuda/std/concepts>
#include <cuda/std/iterator>

static_assert(cuda::std::incrementable<int>);
static_assert(cuda::std::incrementable<int*>);
static_assert(cuda::std::incrementable<int**>);

static_assert(!cuda::std::incrementable<postfix_increment_returns_void>);
static_assert(!cuda::std::incrementable<postfix_increment_returns_copy>);
static_assert(!cuda::std::incrementable<has_integral_minus>);
static_assert(!cuda::std::incrementable<has_distinct_difference_type_and_minus>);
static_assert(!cuda::std::incrementable<missing_difference_type>);
static_assert(!cuda::std::incrementable<floating_difference_type>);
static_assert(!cuda::std::incrementable<non_const_minus>);
static_assert(!cuda::std::incrementable<non_integral_minus>);
static_assert(!cuda::std::incrementable<bad_difference_type_good_minus>);
static_assert(!cuda::std::incrementable<not_default_initializable>);
static_assert(!cuda::std::incrementable<not_movable>);
static_assert(!cuda::std::incrementable<preinc_not_declared>);
static_assert(!cuda::std::incrementable<postinc_not_declared>);
static_assert(cuda::std::incrementable<incrementable_with_difference_type>);
static_assert(cuda::std::incrementable<incrementable_without_difference_type>);
static_assert(cuda::std::incrementable<difference_type_and_void_minus>);
static_assert(!cuda::std::incrementable<noncopyable_with_difference_type>);
static_assert(!cuda::std::incrementable<noncopyable_without_difference_type>);
static_assert(!cuda::std::incrementable<noncopyable_with_difference_type_and_minus>);

int main(int, char**)
{
  return 0;
}
