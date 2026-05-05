//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: dynamic memory allocation is unsupported in tile code

#include <cuda/std/random>

#include "random_utilities/test_engine.h"

TEST_FUNC void test()
{
  test_engine<cuda::std::minstd_rand0, 1043618065u>();
  test_engine<cuda::std::minstd_rand, 399268537ull>();
}

// P4037R1: linear_congruential_engine must accept the unsigned char UIntType
TEST_FUNC void test_p4037r1_small_uinttype()
{
  using E = cuda::std::linear_congruential_engine<unsigned char, 5u, 1u, 13u>;
  static_assert(cuda::std::is_same_v<E::result_type, unsigned char>);
  static_assert(E::min() == 0u);
  static_assert(E::max() == 12u);
  static_assert(E::multiplier == 5u);
  static_assert(E::increment == 1u);
  static_assert(E::modulus == 13u);
}

int main(int, char**)
{
  test();
  test_p4037r1_small_uinttype();
  return 0;
}
