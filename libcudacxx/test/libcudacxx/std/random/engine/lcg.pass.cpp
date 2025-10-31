
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

#include "test_engine.h"
#if !_CCCL_COMPILER(NVRTC)
#  include <random>
#endif // !_CCCL_COMPILER(NVRTC)

__host__ __device__ void test()
{
  test_engine<cuda::std::minstd_rand0, 1043618065u>();
  test_engine<cuda::std::minstd_rand, 399268537ull>();
}

#if !_CCCL_COMPILER(NVRTC)

template <typename E1, typename E2>
void test_std_engines()
{
  int seeds[]    = {E1::default_seed, 12345, 67890, 0, 1, 2147483646};
  int discards[] = {0, 232, 1000};
  for (auto seed : seeds)
  {
    E1 e1(seed);
    E2 e2(seed);
    for (auto d : discards)
    {
      e1.discard(d);
      e2.discard(d);
      for (int i = 0; i < 100; ++i)
      {
        auto v1 = e1();
        auto v2 = e2();
        assert(v1 == v2);
      }
    }
  }
  cuda::std::seed_seq seq({42, 43, 44, 45});
  E1 e1(seq);
  E2 e2(seq);
  for (int i = 0; i < 100; ++i)
  {
    auto v1 = e1();
    auto v2 = e2();
    assert(v1 == v2);
  }
}

void test_against_std()
{
  test_std_engines<cuda::std::minstd_rand0, std::minstd_rand0>();
  test_std_engines<cuda::std::minstd_rand, std::minstd_rand>();
  // 64-bit engine multiplier and increment
  test_std_engines<cuda::std::linear_congruential_engine<cuda::std::uint64_t, 1ul << 32, 1ul << 31, 1ul << 50>,
                   std::linear_congruential_engine<std::uint64_t, 1ul << 32, 1ul << 31, 1ul << 50>>();
}

#endif // !_CCCL_COMPILER(NVRTC

int main(int, char**)
{
  test();
  NV_IF_TARGET(NV_IS_HOST, ({ test_against_std(); }));
  return 0;
}
