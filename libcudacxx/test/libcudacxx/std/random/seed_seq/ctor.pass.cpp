//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  ::cuda::std::array<::cuda::std::uint32_t, 3> seeds_copy{};
  ::cuda::std::array<::cuda::std::uint32_t, 3> seeds = {1, 2, 3};
  // Iterator constructor
  ::cuda::std::seed_seq seq1(seeds.begin(), seeds.end());
  assert(seq1.size() == 3);
  seq1.param(seeds_copy.begin());
  assert(seeds_copy == seeds);
  // Default constructor
  ::cuda::std::seed_seq seq2{};
  assert(seq2.size() == 0);
  ::cuda::std::array<::cuda::std::uint32_t, 0> empty_seeds_copy{};
  seq2.param(empty_seeds_copy.begin());
  assert(empty_seeds_copy.empty());

  // Initializer list constructor
  ::cuda::std::seed_seq seq3{4, 5, 6, 7, 8};
  assert(seq3.size() == 5);
  ::cuda::std::array<::cuda::std::uint32_t, 5> init_seeds_copy{};
  seq3.param(init_seeds_copy.begin());
  ::cuda::std::array<::cuda::std::uint32_t, 5> reference = {4, 5, 6, 7, 8};
  assert(init_seeds_copy == reference);
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif
  return 0;
}
