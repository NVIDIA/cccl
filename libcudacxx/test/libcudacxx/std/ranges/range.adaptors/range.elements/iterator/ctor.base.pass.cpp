//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr explicit iterator(iterator_t<Base> current);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"

// Test explicit
using BaseIter = cuda::std::tuple<int>*;
using ElementsIter =
  cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<cuda::std::ranges::subrange<BaseIter, BaseIter>, 0>>;

static_assert(cuda::std::is_constructible_v<ElementsIter, BaseIter>);
static_assert(!cuda::std::is_convertible_v<BaseIter, ElementsIter>);

struct TracedMoveIter : IterBase<TracedMoveIter>
{
  bool moved = false;

  constexpr TracedMoveIter()                      = default;
  constexpr TracedMoveIter(const TracedMoveIter&) = default;
  __host__ __device__ constexpr TracedMoveIter(TracedMoveIter&&)
      : moved{true}
  {}
  constexpr TracedMoveIter& operator=(TracedMoveIter&&)      = default;
  constexpr TracedMoveIter& operator=(const TracedMoveIter&) = default;
};

struct TracedMoveView : cuda::std::ranges::view_base
{
  __host__ __device__ TracedMoveIter begin() const
  {
    return TracedMoveIter{};
  }
  __host__ __device__ TracedMoveIter end() const
  {
    return TracedMoveIter{};
  }
};

__host__ __device__ constexpr bool test()
{
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<TracedMoveView, 0>>;
  Iter iter{TracedMoveIter{}};
  assert(iter.base().moved);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
