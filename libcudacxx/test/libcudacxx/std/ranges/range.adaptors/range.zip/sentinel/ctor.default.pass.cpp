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

struct PODSentinel
{
  bool b; // deliberately uninitialised

  __host__ __device__ friend constexpr bool operator==(int*, const PODSentinel& s)
  {
    return s.b;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const PODSentinel& s, int*)
  {
    return s.b;
  }
  __host__ __device__ friend constexpr bool operator!=(int*, const PODSentinel& s)
  {
    return !s.b;
  }
  __host__ __device__ friend constexpr bool operator!=(const PODSentinel& s, int*)
  {
    return !s.b;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
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
  {
    using R        = cuda::std::ranges::zip_view<Range>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;
    static_assert(!cuda::std::is_same_v<Sentinel, cuda::std::ranges::iterator_t<R>>);

    cuda::std::ranges::iterator_t<R> it;

    Sentinel s1;
    assert(it != s1); // PODSentinel.b is initialised to false

    Sentinel s2 = {};
    assert(it != s2); // PODSentinel.b is initialised to false
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
