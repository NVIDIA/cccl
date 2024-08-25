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

// constexpr explicit elements_view(V base);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"

struct View : cuda::std::ranges::view_base
{
  MoveOnly mo;
  __host__ __device__ cuda::std::tuple<int>* begin() const;
  __host__ __device__ cuda::std::tuple<int>* end() const;
};

// Test Explicit
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::elements_view<View, 0>, View>);
static_assert(!cuda::std::is_convertible_v<View, cuda::std::ranges::elements_view<View, 0>>);

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::elements_view<View, 0> ev{View{{}, MoveOnly{5}}};
    assert(cuda::std::move(ev).base().mo.get() == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
