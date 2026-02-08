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
__host__ __device__ constexpr void compare(__uint128_t x, cuda::__pcg_uint128_fallback b)
{
  assert(static_cast<::cuda::std::uint64_t>(b) == static_cast<::cuda::std::uint64_t>(x));
  assert(static_cast<::cuda::std::uint64_t>(b >> 64) == static_cast<::cuda::std::uint64_t>(x >> 64));
}

__host__ __device__ constexpr bool test_fallback_uint128(__uint128_t a, __uint128_t b)
{
  using Fallback = cuda::__pcg_uint128_fallback;

  Fallback a_(a >> 64, static_cast<::cuda::std::uint64_t>(a));
  Fallback b_(b >> 64, static_cast<::cuda::std::uint64_t>(b));

  // Test bitwise OR
  {
    compare(a | static_cast<cuda::std::uint64_t>(b), a_ | static_cast<::cuda::std::uint64_t>(b));
  }

  // Test bitwise XOR
  {
    compare(a ^ b, a_ ^ b_);
  }

  // Test left shift
  {
    compare(b << 32, b_ << 32);
    compare(b << 1, b_ << 1);
    compare(b << 0, b_ << 0);
    compare(b << 127, b_ << 127);
  }

  // Test right shift
  {
    compare(b >> 16, b_ >> 16);
    compare(b >> 1, b_ >> 1);
    compare(b >> 0, b_ >> 0);
    compare(b >> 127, b_ >> 127);
  }

  // Test addition
  {
    compare(a + b, a_ + b_);
  }

  // Test multiplication
  {
    compare(a * b, a_ * b_);
  }

  // Test comparison
  {
    assert((a_ == a_) == true);
    assert((a_ == b_) == false);
    assert((a_ != b_) == true);
  }

  return true;
}
__host__ __device__ constexpr bool test_fallback()
{
  // Generate 100 different test values using PCG engine
  cuda::pcg64 rng(42);

  for (int i = 0; i < 100; ++i)
  {
    // Generate two random 128-bit values
    __uint128_t a = static_cast<__uint128_t>(rng()) | (static_cast<__uint128_t>(rng()) << 64);
    __uint128_t b = static_cast<__uint128_t>(rng()) | (static_cast<__uint128_t>(rng()) << 64);

    test_fallback_uint128(a, b);
  }

  return true;
}

#endif // _CCCL_HAS_INT128()

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

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test_fallback();
#endif // _CCCL_HAS_INT128()

  test_engine<cuda::pcg64, 11135645891219275043ul>();
  test_against_reference();
  return 0;
}
