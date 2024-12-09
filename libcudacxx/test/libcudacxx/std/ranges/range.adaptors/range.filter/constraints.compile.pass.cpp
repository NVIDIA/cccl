//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

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

#if TEST_STD_VER >= 2020
template <class View, class Pred>
concept can_form_filter_view = requires { typename cuda::std::ranges::filter_view<View, Pred>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class View, class Pred, class = void>
inline constexpr bool can_form_filter_view = false;

template <class View, class Pred>
inline constexpr bool
  can_form_filter_view<View, Pred, cuda::std::void_t<typename cuda::std::ranges::filter_view<View, Pred>>> = true;
#endif // TEST_STD_VER <= 2017

// filter_view is not valid when the view is not an input_range
namespace test1
{
struct View : cuda::std::ranges::view_base
{
  struct NotInputIterator
  {
    __host__ __device__ NotInputIterator& operator++();
    __host__ __device__ void operator++(int);
    __host__ __device__ int& operator*() const;
    using difference_type = cuda::std::ptrdiff_t;
    __host__ __device__ friend bool operator==(NotInputIterator const&, NotInputIterator const&);
#if TEST_STD_VER <= 2017
    __host__ __device__ friend bool operator!=(NotInputIterator const&, NotInputIterator const&);
#endif // TEST_STD_VER <= 2017
  };
  __host__ __device__ NotInputIterator begin() const;
  __host__ __device__ NotInputIterator end() const;
};
struct Pred
{
  __host__ __device__ bool operator()(int) const;
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
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
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
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
struct Pred
{
  __host__ __device__ bool operator()(int) const;
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
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
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
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
struct Pred
{
  __host__ __device__ bool operator()(int) const;
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
