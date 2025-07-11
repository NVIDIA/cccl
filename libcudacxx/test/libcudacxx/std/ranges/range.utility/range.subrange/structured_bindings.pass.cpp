//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr void test_sized_subrange()
{
  int a[4]      = {1, 2, 3, 4};
  auto r        = cuda::std::ranges::subrange<int*>(a, a + 4);
  const auto cr = cuda::std::ranges::subrange<int*>(a, a + 4);
  assert(cuda::std::ranges::sized_range<decltype(r)>);
  {
    auto [first, last] = r;
    assert(first == a);
    assert(last == a + 4);
  }
  {
    auto [first, last] = cuda::std::move(r);
    assert(first == a);
    assert(last == a + 4);
  }
  {
    auto [first, last] = cr;
    assert(first == a);
    assert(last == a + 4);
  }
  {
    auto [first, last] = cuda::std::move(cr);
    assert(first == a);
    assert(last == a + 4);
  }
}

__host__ __device__ constexpr void test_unsized_subrange()
{
  int a[4] = {1, 2, 3, 4};
  auto r   = cuda::std::ranges::subrange<int*, cuda::std::unreachable_sentinel_t>(a, cuda::std::unreachable_sentinel);
  const auto cr =
    cuda::std::ranges::subrange<int*, cuda::std::unreachable_sentinel_t>(a, cuda::std::unreachable_sentinel);
  assert(!cuda::std::ranges::sized_range<decltype(r)>);
  {
    auto [first, last] = r;
    assert(first == a);
    static_assert(cuda::std::is_same_v<decltype(last), cuda::std::unreachable_sentinel_t>);
  }
  {
    auto [first, last] = cuda::std::move(r);
    assert(first == a);
    static_assert(cuda::std::is_same_v<decltype(last), cuda::std::unreachable_sentinel_t>);
  }
  {
    auto [first, last] = cr;
    assert(first == a);
    static_assert(cuda::std::is_same_v<decltype(last), cuda::std::unreachable_sentinel_t>);
  }
  {
    auto [first, last] = cuda::std::move(cr);
    assert(first == a);
    static_assert(cuda::std::is_same_v<decltype(last), cuda::std::unreachable_sentinel_t>);
  }
}

__host__ __device__ constexpr void test_copies_not_originals()
{
  int a[4] = {1, 2, 3, 4};
  {
    auto r               = cuda::std::ranges::subrange<int*>(a, a + 4);
    auto&& [first, last] = r;
    static_assert(cuda::std::is_same_v<decltype(first), int*>);
    static_assert(cuda::std::is_same_v<decltype(last), int*>);
    first = a + 2;
    last  = a + 2;
    assert(r.begin() == a);
    assert(r.end() == a + 4);
  }
// For reasons unknown nvrtc complains that `__begin_` is not accessible here...
#if !TEST_COMPILER(NVRTC)
  {
    const auto r         = cuda::std::ranges::subrange<int*>(a, a + 4);
    auto&& [first, last] = r;
    static_assert(cuda::std::is_same_v<decltype(first), int*>);
    static_assert(cuda::std::is_same_v<decltype(last), int*>);
    first = a + 2;
    last  = a + 2;
    assert(r.begin() == a);
    assert(r.end() == a + 4);
  }
#endif // !TEST_COMPILER(NVRTC)
}

__host__ __device__ constexpr bool test()
{
  test_sized_subrange();
  test_unsized_subrange();
  test_copies_not_originals();
  return true;
}

int main(int, char**)
{
  test();
#if !TEST_COMPILER(MSVC) // MSVC gives an ICE here
  static_assert(test(), "");
#endif // !TEST_COMPILER(MSVC)

  return 0;
}
