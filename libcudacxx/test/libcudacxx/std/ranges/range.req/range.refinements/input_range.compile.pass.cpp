//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class R>
// concept input_range;

#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(cuda::std::ranges::input_range<test_range<cpp17_input_iterator>>);
static_assert(cuda::std::ranges::input_range<test_range<cpp17_input_iterator> const>);

static_assert(cuda::std::ranges::input_range<test_range<cpp20_input_iterator>>);
static_assert(cuda::std::ranges::input_range<test_range<cpp20_input_iterator> const>);

static_assert(cuda::std::ranges::input_range<test_non_const_range<cpp17_input_iterator>>);
static_assert(cuda::std::ranges::input_range<test_non_const_range<cpp20_input_iterator>>);

static_assert(!cuda::std::ranges::input_range<test_non_const_range<cpp17_input_iterator> const>);
static_assert(!cuda::std::ranges::input_range<test_non_const_range<cpp20_input_iterator> const>);

static_assert(cuda::std::ranges::input_range<test_common_range<forward_iterator>>);
static_assert(!cuda::std::ranges::input_range<test_common_range<cpp20_input_iterator>>);

static_assert(cuda::std::ranges::input_range<test_common_range<forward_iterator> const>);
static_assert(!cuda::std::ranges::input_range<test_common_range<cpp20_input_iterator> const>);

static_assert(cuda::std::ranges::input_range<test_non_const_common_range<forward_iterator>>);
static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<cpp20_input_iterator>>);

static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<forward_iterator> const>);
static_assert(!cuda::std::ranges::input_range<test_non_const_common_range<cpp20_input_iterator> const>);

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*&>);
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>*&&>);
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const>);
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const&>);
static_assert(!cuda::std::ranges::input_range<Holder<Incomplete>* const&&>);

static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* [10]>);
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* (&) [10]>);
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* (&&) [10]>);
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const[10]>);
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const (&)[10]>);
static_assert(cuda::std::ranges::input_range<Holder<Incomplete>* const (&&)[10]>);
#endif

int main(int, char**)
{
  return 0;
}
