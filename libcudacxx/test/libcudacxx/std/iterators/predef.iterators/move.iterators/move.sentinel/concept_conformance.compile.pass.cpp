//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// template<semiregular S>
//   class move_sentinel;
#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"

__host__ __device__ void test()
{
  // Pointer.
  {
    using It = int*;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `Cpp17InputIterator`.
  {
    using It = cpp17_input_iterator<int*>;
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::input_iterator`.
  {
    using It = cpp20_input_iterator<int*>;
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::forward_iterator`.
  {
    using It = forward_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(!cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::bidirectional_iterator`.
  {
    using It = bidirectional_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(!cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::random_access_iterator`.
  {
    using It = random_access_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::contiguous_iterator`.
  {
    using It = contiguous_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  // `cuda::std::contiguous_iterator` with the spaceship operator.
  {
    using It = three_way_contiguous_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }
#endif
}

int main(int, char**)
{
  return 0;
}
