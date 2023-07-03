//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// concept checking
// template<view V, class Pred>
//     requires input_range<V> && is_object_v<Pred> &&
//              indirect_unary_predicate<const Pred, iterator_t<V>>
//   class take_while_view;

#include <cuda/std/array>
#include <cuda/std/ranges>

#include "test_iterators.h"

template <class It>
using Range = cuda::std::ranges::subrange<It, sentinel_wrapper<It>>;

template <class Val = int>
struct Pred {
  __host__ __device__ bool operator()(const Val&) const;
};

#if TEST_STD_VER > 17
template <class V, class Pred>
concept HasTakeWhileView = requires { typename cuda::std::ranges::take_while_view<V, Pred>; };
#else
template <class V, class Pred, class = void>
inline constexpr bool HasTakeWhileView = false;

template <class V, class Pred>
inline constexpr bool HasTakeWhileView<V, Pred,
  cuda::std::void_t<typename cuda::std::ranges::take_while_view<V, Pred>>> = true;
#endif

static_assert(HasTakeWhileView<Range<int*>, bool (*)(int)>);
static_assert(HasTakeWhileView<Range<int*>, Pred<int>>);

// !view<V>
static_assert(!HasTakeWhileView<cuda::std::array<int, 5>, Pred<int>>);

// !input_range
static_assert(!HasTakeWhileView<Range<cpp20_output_iterator<int*>>, bool (*)(int)>);

// !is_object_v<Pred>
static_assert(!HasTakeWhileView<Range<int*>, Pred<int>&>);

// !indirect_unary_predicate<const Pred, iterator_t<V>>
static_assert(!HasTakeWhileView<Range<int*>, int>);
static_assert(!HasTakeWhileView<Range<int**>, Pred<int>>);

int main(int, char**) {
  return 0;
}
