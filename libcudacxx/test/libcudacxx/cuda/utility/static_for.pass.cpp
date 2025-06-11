//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <typename ExpectedType>
struct Op
{
  int i         = 0;
  int step      = 1;
  int max_iters = 0;
  int count     = 0;

  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index)
  {
    static_assert(cuda::std::is_same_v<ExpectedType, decltype(index())>);
    constexpr int value = index(); // compile-time evaluation
    assert(value == i);
    i += step;
    assert(count < max_iters);
    ++count;
  }
};

struct OpArgs
{
  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index, int, int, int)
  {
    [[maybe_unused]] constexpr int value = index(); // compile-time evaluation
  }
};

struct Op2D
{
  template <typename Index>
  __host__ __device__ constexpr void operator()(Index index)
  {
    using index_t = typename Index::value_type;
    if constexpr (index > 0)
    {
      cuda::static_for<index()>(Op<index_t>{0, 1, index()});
      cuda::static_for<index_t, index()>(Op<index_t>{0, 1, index()});
    }
  }
};

template <typename T>
__host__ __device__ constexpr void test()
{
  cuda::static_for<T{10}>(Op<T>{0, 1, 10});
  cuda::static_for<T{10}>(OpArgs{}, 1, 2, 3);
  cuda::static_for<T{10}>(Op2D{});

  cuda::static_for<T{15}, 20>(Op<T>{15, 1, 5});
  cuda::static_for<T{15}, 137, 5>(Op<T>{15, 5, 137 - 15 / 5});

  if constexpr (cuda::std::is_signed_v<T>)
  {
    cuda::static_for<T{-15}, T{15}, T{5}>(Op<T>{-15, 5, 6});
    cuda::static_for<T{15}, T{-15}, T{-5}>(Op<T>{15, -5, 6});
  }
}

__host__ __device__ constexpr bool test()
{
  test<short>();
  test<int>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();
  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
