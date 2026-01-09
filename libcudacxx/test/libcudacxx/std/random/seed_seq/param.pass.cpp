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
  cuda::std::array<cuda::std::uint32_t, 3> seeds = {1, 2, 3};
  cuda::std::seed_seq seq1(seeds.begin(), seeds.end());
  cuda::std::array<cuda::std::uint32_t, 3> seeds_copy{};
  seq1.param(seeds_copy.begin());
  assert(seeds_copy == seeds);
  static_assert(cuda::std::is_void_v<decltype(seq1.param(seeds_copy.begin()))>);
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif
  return 0;
}
