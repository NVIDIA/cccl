//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// concept checking
// template<view V, class Pred>
//     requires input_range<V> && is_object_v<Pred> &&
//              indirect_unary_predicate<const Pred, iterator_t<V>>
//   class drop_while_view;

#include <cuda/std/array>
#include <cuda/std/ranges>

#include "test_iterators.h"

template <class It>
using Range = cuda::std::ranges::subrange<It, sentinel_wrapper<It>>;

template <class Val = int>
struct Pred
{
  __host__ __device__ bool operator()(const Val&) const;
};

template <class View, class Pred>
_CCCL_CONCEPT HasDropWhileView =
  _CCCL_REQUIRES_EXPR((View, Pred), View v)(typename(typename cuda::std::ranges::drop_while_view<View, Pred>));

static_assert(HasDropWhileView<Range<int*>, bool (*)(int)>);
static_assert(HasDropWhileView<Range<int*>, Pred<int>>);

// !view<V>
static_assert(!HasDropWhileView<cuda::std::array<int, 5>, Pred<int>>);

// !input_range
static_assert(!HasDropWhileView<Range<cpp20_output_iterator<int*>>, bool (*)(int)>);

// !is_object_v<Pred>
static_assert(!HasDropWhileView<Range<int*>, Pred<int>&>);

// !indirect_unary_predicate<const Pred, iterator_t<V>>
static_assert(!HasDropWhileView<Range<int*>, int>);
static_assert(!HasDropWhileView<Range<int**>, Pred<int>>);

int main(int, char**)
{
  return 0;
}
