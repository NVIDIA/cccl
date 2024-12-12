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

// drop_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

template <bool DefaultInitializable>
struct View : cuda::std::ranges::view_base
{
  int i                     = 42;
  constexpr explicit View() = default;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

template <>
struct View<false> : cuda::std::ranges::view_base
{
  int i  = 42;
  View() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

template <bool DefaultInitializable>
struct Pred
{
  int i                     = 42;
  constexpr explicit Pred() = default;
  __host__ __device__ bool operator()(int) const;
};
template <>
struct Pred<false>
{
  int i  = 42;
  Pred() = delete;
  __host__ __device__ bool operator()(int) const;
};

// clang-format off
static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::drop_while_view<View<true >, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::drop_while_view<View<false>, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::drop_while_view<View<true >, Pred<false>>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::drop_while_view<View<false>, Pred<false>>>);
// clang-format on

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::drop_while_view<View<true>, Pred<true>> dwv = {};
    assert(dwv.base().i == 42);
    assert(dwv.pred().i == 42);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020
  return 0;
}
