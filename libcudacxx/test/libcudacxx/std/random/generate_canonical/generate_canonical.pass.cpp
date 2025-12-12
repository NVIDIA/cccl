//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//

#include <cuda/std/__random_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/numeric>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  cuda::std::array<T, 100> data;
  cuda::std::philox4x64 g{};
  for (auto& v : data)
  {
    v = cuda::std::generate_canonical<T, cuda::std::numeric_limits<T>::digits>(g);
    assert(v >= T(0) && v < T(1));
  }
  auto mean = cuda::std::accumulate(data.begin(), data.end(), T(0)) / data.size();
  assert(mean > T(0.4) && mean < T(0.6));
}

int main(int, char**)
{
  test<float>();
  test<double>();
  return 0;
}
