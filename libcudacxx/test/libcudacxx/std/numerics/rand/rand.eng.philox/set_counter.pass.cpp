
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/std/random>
template <typename Engine>
__host__ __device__ void test()
{
  Engine e1(7);
  Engine e2(7);
  e1.discard(100);
  e2.set_counter({0, 0, 0, 100 / 4});
  assert(e1 == e2);

  // Set counter can go "back" to where it was before
  e1 = Engine(7);
  e2.set_counter({0, 0, 0, 0});
  assert(e1 == e2);
  // Overflow the counter
  e1 = Engine(7);
  e2 = Engine(7);
  e1.set_counter({0, 0, 1, 0});
  if constexpr (std::is_same_v<Engine, cuda::std::philox4x32>)
  {
    e2.set_counter({0, 0, 0, cuda::std::numeric_limits<std::uint32_t>::max()});
  }
  else
  {
    e2.set_counter({0, 0, 0, cuda::std::numeric_limits<std::uint64_t>::max()});
  }
  e2.discard(4);
  assert(e1 == e2);
}

int main(int, char**)
{
  test<cuda::std::philox4x32>();
  // test<cuda::std::philox4x64>();
  return 0;
}
