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

// constexpr sentinel_t<Base> base() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct Sent
{
  int i;

  __host__ __device__ friend constexpr bool operator==(cuda::std::tuple<int>*, const Sent&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const Sent&, cuda::std::tuple<int>*)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(cuda::std::tuple<int>*, const Sent&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

__host__ __device__ constexpr bool test()
{
  using BaseRange = cuda::std::ranges::subrange<cuda::std::tuple<int>*, Sent>;
  using EleRange  = cuda::std::ranges::elements_view<BaseRange, 0>;
  using EleSent   = cuda::std::ranges::sentinel_t<EleRange>;

  const EleSent st{Sent{5}};
  decltype(auto) base = st.base();
  static_assert(cuda::std::same_as<decltype(base), Sent>);
  assert(base.i == 5);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
