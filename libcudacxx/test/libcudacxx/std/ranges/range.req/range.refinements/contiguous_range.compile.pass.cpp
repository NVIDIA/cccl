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
// concept contiguous_range;

#include <cuda/std/ranges>

#include "test_range.h"
#include "test_iterators.h"

namespace ranges = cuda::std::ranges;

template <template <class...> class I>
TEST_HOST_DEVICE constexpr bool check_range() {
  constexpr bool result = ranges::contiguous_range<test_range<I> >;
  static_assert(ranges::contiguous_range<test_range<I> const> == result);
  static_assert(ranges::contiguous_range<test_non_const_common_range<I> > == result);
  static_assert(ranges::contiguous_range<test_non_const_range<I> > == result);
  static_assert(ranges::contiguous_range<test_common_range<I> > == result);
  static_assert(ranges::contiguous_range<test_common_range<I> const> == result);
  static_assert(!ranges::contiguous_range<test_non_const_common_range<I> const>);
  static_assert(!ranges::contiguous_range<test_non_const_range<I> const>);
  return result;
}

static_assert(!check_range<cpp20_input_iterator>());
static_assert(!check_range<forward_iterator>());
static_assert(!check_range<bidirectional_iterator>());
static_assert(!check_range<random_access_iterator>());
static_assert(check_range<contiguous_iterator>());

struct ContiguousWhenNonConst {
    TEST_HOST_DEVICE const int *begin() const;
    TEST_HOST_DEVICE const int *end() const;
    TEST_HOST_DEVICE int *begin();
    TEST_HOST_DEVICE int *end();
    TEST_HOST_DEVICE int *data() const;
};
static_assert( cuda::std::ranges::contiguous_range<ContiguousWhenNonConst>);
static_assert( cuda::std::ranges::random_access_range<const ContiguousWhenNonConst>);
static_assert(!cuda::std::ranges::contiguous_range<const ContiguousWhenNonConst>);

struct ContiguousWhenConst {
    TEST_HOST_DEVICE const int *begin() const;
    TEST_HOST_DEVICE const int *end() const;
    TEST_HOST_DEVICE int *begin();
    TEST_HOST_DEVICE int *end();
    TEST_HOST_DEVICE const int *data() const;
};
static_assert( cuda::std::ranges::contiguous_range<const ContiguousWhenConst>);
static_assert( cuda::std::ranges::random_access_range<ContiguousWhenConst>);
static_assert(!cuda::std::ranges::contiguous_range<ContiguousWhenConst>);

struct DataFunctionWrongReturnType {
    TEST_HOST_DEVICE const int *begin() const;
    TEST_HOST_DEVICE const int *end() const;
    TEST_HOST_DEVICE const char *data() const;
};
static_assert( cuda::std::ranges::random_access_range<DataFunctionWrongReturnType>);
static_assert(!cuda::std::ranges::contiguous_range<DataFunctionWrongReturnType>);

struct WrongObjectness {
    TEST_HOST_DEVICE const int *begin() const;
    TEST_HOST_DEVICE const int *end() const;
    TEST_HOST_DEVICE void *data() const;
};
static_assert(cuda::std::ranges::contiguous_range<WrongObjectness>);

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };

static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>*&>);
static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>*&&>);
static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>* const>);
static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>* const&>);
static_assert(!cuda::std::ranges::contiguous_range<Holder<Incomplete>* const&&>);

static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>*[10]>);
static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>*(&)[10]>);
static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>*(&&)[10]>);
static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>* const[10]>);
static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>* const(&)[10]>);
static_assert( cuda::std::ranges::contiguous_range<Holder<Incomplete>* const(&&)[10]>);
#endif

int main(int, char**) {
  return 0;
}
