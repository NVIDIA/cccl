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

// take_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

template <bool defaultInitable>
struct View : cuda::std::ranges::view_base
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  __host__ __device__ constexpr explicit View() noexcept {};
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

template <bool defaultInitable>
struct Pred
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  __host__ __device__ constexpr explicit Pred() noexcept {};
  __host__ __device__ bool operator()(int) const;
};

// clang-format off
static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<false>>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<false>>>);
// clang-format on

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_while_view<View<true>, Pred<true>> twv = {};
    assert(twv.base().i == 0);
    assert(twv.pred().i == 0);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
