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

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_macros.h"

struct PODSentinel
{
  int i; // deliberately uninitialised

  __host__ __device__ friend constexpr bool operator==(cuda::std::tuple<int>*, const PODSentinel&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const PODSentinel&, cuda::std::tuple<int>*)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(cuda::std::tuple<int>*, const PODSentinel&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const PODSentinel&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ cuda::std::tuple<int>* begin() const
  {
    return nullptr;
  }
  __host__ __device__ PODSentinel end()
  {
    return PODSentinel{};
  }
};

__host__ __device__ constexpr bool test()
{
  using EleView  = cuda::std::ranges::elements_view<Range, 0>;
  using Sentinel = cuda::std::ranges::sentinel_t<EleView>;
  static_assert(!cuda::std::is_same_v<Sentinel, cuda::std::ranges::iterator_t<EleView>>);

  {
    Sentinel s;
    assert(s.base().i == 0);
  }
  {
    Sentinel s = {};
    assert(s.base().i == 0);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
