//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr const Pred& pred() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct View : cuda::std::ranges::view_interface<View>
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  int i;
  __host__ __device__ bool operator()(int) const;
};

__host__ __device__ constexpr bool test()
{
  // &
  {
    cuda::std::ranges::take_while_view<View, Pred> twv{{}, Pred{5}};
    decltype(auto) x = twv.pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &
  {
    const cuda::std::ranges::take_while_view<View, Pred> twv{{}, Pred{5}};
    decltype(auto) x = twv.pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // &&
  {
    cuda::std::ranges::take_while_view<View, Pred> twv{{}, Pred{5}};
    decltype(auto) x = cuda::std::move(twv).pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &&
  {
    const cuda::std::ranges::take_while_view<View, Pred> twv{{}, Pred{5}};
    decltype(auto) x = cuda::std::move(twv).pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
