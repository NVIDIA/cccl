//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Check constraints on the type itself.
//
// template<input_range View, indirect_unary_predicate<iterator_t<View>> Pred>
//    requires view<View> && is_object_v<Pred>
// class filter_view;

#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class View, class Pred>
_CCCL_CONCEPT can_form_filter_view =
  _CCCL_REQUIRES_EXPR((View, Pred), )(typename(cuda::std::ranges::filter_view<View, Pred>));

// filter_view is not valid when the view is not an input_range
namespace test1
{
struct View : cuda::std::ranges::view_base
{
  struct NotInputIterator
  {
    TEST_FUNC NotInputIterator& operator++();
    TEST_FUNC void operator++(int);
    TEST_FUNC int& operator*() const;
    using difference_type = cuda::std::ptrdiff_t;
    TEST_FUNC friend bool operator==(NotInputIterator const&, NotInputIterator const&);
    TEST_FUNC friend bool operator!=(NotInputIterator const&, NotInputIterator const&);
  };
  TEST_FUNC NotInputIterator begin() const;
  TEST_FUNC NotInputIterator end() const;
};
struct Pred
{
  TEST_FUNC bool operator()(int) const;
};

static_assert(!cuda::std::ranges::input_range<View>);
static_assert(cuda::std::indirect_unary_predicate<Pred, int*>);
static_assert(cuda::std::ranges::view<View>);
static_assert(cuda::std::is_object_v<Pred>);
static_assert(!can_form_filter_view<View, Pred>);
} // namespace test1

// filter_view is not valid when the predicate is not indirect_unary_predicate
namespace test2
{
struct View : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
struct Pred
{};

static_assert(cuda::std::ranges::input_range<View>);
static_assert(!cuda::std::indirect_unary_predicate<Pred, int*>);
static_assert(cuda::std::ranges::view<View>);
static_assert(cuda::std::is_object_v<Pred>);
static_assert(!can_form_filter_view<View, Pred>);
} // namespace test2

// filter_view is not valid when the view is not a view
namespace test3
{
struct View
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
struct Pred
{
  TEST_FUNC bool operator()(int) const;
};

static_assert(cuda::std::ranges::input_range<View>);
static_assert(cuda::std::indirect_unary_predicate<Pred, int*>);
static_assert(!cuda::std::ranges::view<View>);
static_assert(cuda::std::is_object_v<Pred>);
static_assert(!can_form_filter_view<View, Pred>);
} // namespace test3

// filter_view is not valid when the predicate is not an object type
namespace test4
{
struct View : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
using Pred = bool (&)(int);

static_assert(cuda::std::ranges::input_range<View>);
static_assert(cuda::std::indirect_unary_predicate<Pred, int*>);
static_assert(cuda::std::ranges::view<View>);
static_assert(!cuda::std::is_object_v<Pred>);
static_assert(!can_form_filter_view<View, Pred>);
} // namespace test4

// filter_view is valid when all the constraints are satisfied (test the test)
namespace test5
{
struct View : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
struct Pred
{
  TEST_FUNC bool operator()(int) const;
};

static_assert(cuda::std::ranges::input_range<View>);
static_assert(cuda::std::indirect_unary_predicate<Pred, int*>);
static_assert(cuda::std::ranges::view<View>);
static_assert(cuda::std::is_object_v<Pred>);
static_assert(can_form_filter_view<View, Pred>);
} // namespace test5

int main(int, char**)
{
  return 0;
}
