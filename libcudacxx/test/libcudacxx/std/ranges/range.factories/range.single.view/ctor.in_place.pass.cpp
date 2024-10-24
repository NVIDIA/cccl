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

// template<class... Args>
//   requires constructible_from<T, Args...>
// constexpr explicit single_view(in_place_t, Args&&... args);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct TakesTwoInts
{
  int a_, b_;
  __host__ __device__ constexpr TakesTwoInts(int a, int b)
      : a_(a)
      , b_(b)
  {}
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::single_view<TakesTwoInts> sv(cuda::std::in_place, 1, 2);
    assert(sv.data()->a_ == 1);
    assert(sv.data()->b_ == 2);
    assert(sv.size() == 1);
  }
  {
    const cuda::std::ranges::single_view<TakesTwoInts> sv(cuda::std::in_place, 1, 2);
    assert(sv.data()->a_ == 1);
    assert(sv.data()->b_ == 2);
    assert(sv.size() == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
