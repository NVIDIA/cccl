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

#include "random_utilities/test_engine.h"

template <typename Engine>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_set_counter()
{
  Engine e1(7);
  Engine e2(7);
  e1.discard(100);
  static_assert(::cuda::std::is_void_v<decltype(e2.set_counter({0, 0, 0, 100}))>);
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
  if constexpr (::cuda::std::is_same_v<Engine, typename ::cuda::std::philox4x32>)
  {
    e2.set_counter({0, 0, 0, ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max()});
  }
  else
  {
    e2.set_counter({0, 0, 0, ::cuda::std::numeric_limits<::cuda::std::uint64_t>::max()});
  }
  e2.discard(4);
  assert(e1 == e2);
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_against_reference()
{
  // reference values obtained from other standard library implementations
  const int seeds[]                               = {10823018, 0, 23};
  const int discards[]                            = {0, 5, 100};
  const cuda::std::uint64_t reference_values_64[] = {
    597860052874975753ull,
    16480731654955167298ull,
    4676222276634405366ull,
    1609277786247541068ull,
    4455796210202625458ull,
    12591023382997339072ull,
    13350274857560636235ull,
    5430341342746607840ull,
    4983921884708484958ull};
  const cuda::std::uint32_t reference_values_32[] = {
    1514423753u, 254961463u, 4167151386u, 1713891541u, 1555169499u, 444026393u, 1368340107u, 2016696101u, 1090885419u};

  int ref_index = 0;
  for (auto seed : seeds)
  {
    for (auto discard : discards)
    {
      cuda::std::philox4x64 rng(seed);
      cuda::std::philox4x32 rng32(seed);
      rng.discard(discard);
      rng32.discard(discard);
      assert(rng() == reference_values_64[ref_index]);
      assert(rng32() == reference_values_32[ref_index]);
      ref_index++;
    }
  }
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  test_engine<cuda::std::philox4x32, 1955073260u>();
  test_engine<cuda::std::philox4x64, 3409172418970261260ull>();
  test_set_counter<cuda::std::philox4x32>();
  test_set_counter<cuda::std::philox4x64>();
  test_against_reference();
#if TEST_STD_VER >= 2020
  static_assert(test_set_counter<cuda::std::philox4x32>());
  static_assert(test_set_counter<cuda::std::philox4x64>());
  static_assert(test_against_reference());
#endif
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
