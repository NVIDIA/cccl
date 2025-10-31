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

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  cuda::std::array<cuda::std::uint32_t, 3> seeds{1, 2, 3};

  // 1. Default constructor
  {
    cuda::std::seed_seq seq;
    assert(seq.size() == 0);
  }

  // 2. Iterator constructor
  {
    cuda::std::seed_seq seq(seeds.begin(), seeds.end());
    assert(seq.size() == 3);

    cuda::std::array<cuda::std::uint32_t, 3> seeds_copy{};
    seq.param(seeds_copy.begin());
    assert(seeds_copy == seeds);
  }
  // 3. Initializer list constructor
  {
    cuda::std::seed_seq seq{4, 5, 6, 7, 8};
    assert(seq.size() == 5);

    cuda::std::array<cuda::std::uint32_t, 5> init_seeds_copy{};
    seq.param(init_seeds_copy.begin());
    assert((init_seeds_copy == cuda::std::array<cuda::std::uint32_t, 5>{4, 5, 6, 7, 8}));
  }
  // 4. InputIterator constructor
  {
    cuda::std::seed_seq seq(cpp17_input_iterator<cuda::std::uint32_t*>(seeds.begin()),
                            cpp17_input_iterator<cuda::std::uint32_t*>(seeds.end()));
    assert(seq.size() == 3);
    cuda::std::array<cuda::std::uint32_t, 3> seeds_copy{};
    seq.param(seeds_copy.begin());
    assert(seeds_copy == seeds);
  }

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
