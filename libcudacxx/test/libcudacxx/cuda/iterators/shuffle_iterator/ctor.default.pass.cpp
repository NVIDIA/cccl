//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<W> = default;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    [[maybe_unused]] cuda::shuffle_iterator iter;
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<size_t, cuda::random_bijection<size_t, cuda::__feistel_bijection>>>);
  }

  {
    [[maybe_unused]] cuda::shuffle_iterator iter{};
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<size_t, cuda::random_bijection<size_t, cuda::__feistel_bijection>>>);
  }

  {
    [[maybe_unused]] cuda::shuffle_iterator<int> iter;
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<int, cuda::random_bijection<int, cuda::__feistel_bijection>>>);
  }

  {
    [[maybe_unused]] cuda::shuffle_iterator<int> iter{};
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<int, cuda::random_bijection<int, cuda::__feistel_bijection>>>);
  }

  {
    [[maybe_unused]] cuda::shuffle_iterator<int, cuda::random_bijection<size_t>> iter;
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<int, cuda::random_bijection<size_t, cuda::__feistel_bijection>>>);
  }

  {
    [[maybe_unused]] cuda::shuffle_iterator<int, cuda::random_bijection<size_t>> iter{};
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::shuffle_iterator<int, cuda::random_bijection<size_t, cuda::__feistel_bijection>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
