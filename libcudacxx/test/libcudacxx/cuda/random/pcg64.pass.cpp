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

#include "random_utilities/test_engine.h"

#if _CCCL_HAS_INT128()

__host__ __device__ constexpr bool test_against_reference()
{
  // reference values obtained from other library implementations
  constexpr int seeds[]                            = {10823018, 0, 23};
  constexpr int discards[]                         = {0, 5, 100};
  constexpr cuda::std::uint64_t reference_values[] = {
    11492238902574317825ull,
    8322011739913317518ull,
    16162292887622315191ull,
    74029666500212977ull,
    7381380909356947872ull,
    13353295228484708474ull,
    11051782693829522167ull,
    8996870419832475944ull,
    14156256770140333413ull};

  int ref_index = 0;
  for (auto seed : seeds)
  {
    for (auto discard : discards)
    {
      cuda::pcg64 rng(seed);
      rng.discard(discard);
      assert(rng() == reference_values[ref_index]);
      ref_index++;
    }
  }
  return true;
}
#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test_engine<cuda::pcg64, 11135645891219275043ul>();
  test_against_reference();
#endif // _CCCL_HAS_INT128()
  return 0;
}
