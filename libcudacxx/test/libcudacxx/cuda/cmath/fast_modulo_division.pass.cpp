//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/__random_>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

template <typename value_t, typename divisor_t>
__host__ __device__ void test()
{
  constexpr auto max_value = cuda::std::numeric_limits<divisor_t>::max();
  constexpr auto range     = max_value < 1000000 ? max_value : 1000000;
  cuda::std::minstd_rand0 rng;
  cuda::std::uniform_int_distribution<value_t> distrib;
  value_t value = distrib(rng);
  for (divisor_t i = 1; i < range; ++i)
  {
    cuda::fast_mod_div<divisor_t> div_mod(i);
    assert(value / div_mod == value / i);
  }
  cuda::fast_mod_div<divisor_t> div_mod(max_value);
  assert(value / div_mod == value / max_value);
}

__host__ __device__ bool test()
{
  test<int16_t, int16_t>();
  test<uint16_t, uint16_t>();
  test<int, int>();
  test<unsigned, unsigned>();
  //
  // test<int, int16_t>();
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
