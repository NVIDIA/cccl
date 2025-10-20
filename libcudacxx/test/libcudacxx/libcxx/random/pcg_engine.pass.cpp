
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/__random/pcg_engine.h>

#include <random>

template <typename Engine, typename Engine::result_type value_10000>
__host__ __device__ constexpr bool test()
{
  Engine e;
  for (int i = 0; i < 100; ++i)
  {
    Engine e2;
    e2.discard(i);
    assert(e == e2);
    e();
  }

  e = Engine();
  e.discard(9999);
  assert(e() == value_10000);

  return true;
}

class SeedSeq
{
public:
  using result_type = uint32_t;
  template <typename It>
  __host__ __device__ void generate(It begin, It end)
  {
    uint32_t value = 0;
    for (It it = begin; it != end; ++it)
    {
      *it = value++;
    }
  }
};

__host__ __device__ bool test_seed_sequence()
{
  SeedSeq seq;
  // Seed the engine with the seed sequence
  cuda::pcg64_engine e(seq);
  // Value taken from pcg reference implementation
  assert(e() == 6292233566619932430ul);
  return true;
}

int main(int, char**)
{
  test<cuda::pcg64_engine, 11135645891219275043ul>();

  test_seed_sequence();
  // static_assert(test<cuda::pcg64_engine, 11135645891219275043ul>());
  return 0;
}
