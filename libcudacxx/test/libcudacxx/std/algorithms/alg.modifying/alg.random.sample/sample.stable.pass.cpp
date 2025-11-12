//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class PopulationIterator, class SampleIterator, class Distance,
//           class UniformRandomNumberGenerator>
// SampleIterator sample(PopulationIterator first, PopulationIterator last,
//                       SampleIterator out, Distance n,
//                       UniformRandomNumberGenerator &&g);

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

// Stable if and only if PopulationIterator meets the requirements of a
// ForwardIterator type.
template <class PopulationIterator, class SampleIterator>
__host__ __device__ void test_stability(bool expect_stable)
{
  const unsigned kPopulationSize = 100;
  int ia[kPopulationSize];
  for (unsigned i = 0; i < kPopulationSize; ++i)
  {
    ia[i] = i;
  }
  PopulationIterator first(ia);
  PopulationIterator last(ia + kPopulationSize);

  const unsigned kSampleSize = 20;
  int oa[kPopulationSize];
  SampleIterator out(oa);

  cuda::std::minstd_rand g;

  const int kIterations = 1000;
  bool unstable         = false;
  for (int i = 0; i < kIterations; ++i)
  {
    cuda::std::sample(first, last, out, kSampleSize, g);
    unstable |= !cuda::std::is_sorted(oa, oa + kSampleSize);
  }
  assert(expect_stable == !unstable);
}

int main(int, char**)
{
  test_stability<forward_iterator<int*>, cpp17_output_iterator<int*>>(true);
  test_stability<cpp17_input_iterator<int*>, random_access_iterator<int*>>(false);

  return 0;
}
