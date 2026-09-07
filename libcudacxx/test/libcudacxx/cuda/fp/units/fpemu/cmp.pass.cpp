// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu comparison operations vs native IEEE-754 double.
//
//  Validates that the fpemu comparison primitives (==, !=, <, <=, >, >=) match
//  native double comparisons for every value class (normals, subnormals, +/-0,
//  +/-inf, quiet/signaling NaN, unordered cases). Three surfaces are cross-checked
//  against native double: the C builtins (__fp64emu_cmp_*), the packed operators
//  and the unpacked operators.
//
//  Every ordered pair of the special values is checked exhaustively, then a
//  deterministic pseudo-random sweep covers arbitrary bit patterns.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/random>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Comparison operation indices (also bit positions in the packed result code).
enum cmp_op
{
  OP_EQ = 0,
  OP_NE,
  OP_LT,
  OP_LE,
  OP_GT,
  OP_GE,
  OP_COUNT
};

TEST_HOST_DEVICE_FUNC double from_bits(uint64_t b)
{
  return cuda::std::bit_cast<double>(b);
}

// Pack the six native comparison results into one bit code.
TEST_HOST_DEVICE_FUNC uint32_t native_codes(double x, double y)
{
  uint32_t c = 0;
  c |= (uint32_t) (x == y) << OP_EQ;
  c |= (uint32_t) (x != y) << OP_NE;
  c |= (uint32_t) (x < y) << OP_LT;
  c |= (uint32_t) (x <= y) << OP_LE;
  c |= (uint32_t) (x > y) << OP_GT;
  c |= (uint32_t) (x >= y) << OP_GE;
  return c;
}

// Compare all three emulation surfaces against native for one pair.
TEST_HOST_DEVICE_FUNC void check_pair(double x, double y)
{
  const uint32_t ref = native_codes(x, y);

  cudax::__fpbits64 ex = cudax::__fp64emu_from_double(x);
  cudax::__fpbits64 ey = cudax::__fp64emu_from_double(y);
  uint32_t cb          = 0;
  cb |= (uint32_t) cudax::__fp64emu_cmp_eq(ex, ey) << OP_EQ;
  cb |= (uint32_t) cudax::__fp64emu_cmp_ne(ex, ey) << OP_NE;
  cb |= (uint32_t) cudax::__fp64emu_cmp_lt(ex, ey) << OP_LT;
  cb |= (uint32_t) cudax::__fp64emu_cmp_le(ex, ey) << OP_LE;
  cb |= (uint32_t) cudax::__fp64emu_cmp_gt(ex, ey) << OP_GT;
  cb |= (uint32_t) cudax::__fp64emu_cmp_ge(ex, ey) << OP_GE;

  cudax::fp64emu px = x, py = y;
  uint32_t cp = 0;
  cp |= (uint32_t) (px == py) << OP_EQ;
  cp |= (uint32_t) (px != py) << OP_NE;
  cp |= (uint32_t) (px < py) << OP_LT;
  cp |= (uint32_t) (px <= py) << OP_LE;
  cp |= (uint32_t) (px > py) << OP_GT;
  cp |= (uint32_t) (px >= py) << OP_GE;

  cudax::fp64emu_unpacked ux = (cudax::fp64emu_unpacked) x, uy = (cudax::fp64emu_unpacked) y;
  uint32_t cu = 0;
  cu |= (uint32_t) (ux == uy) << OP_EQ;
  cu |= (uint32_t) (ux != uy) << OP_NE;
  cu |= (uint32_t) (ux < uy) << OP_LT;
  cu |= (uint32_t) (ux <= uy) << OP_LE;
  cu |= (uint32_t) (ux > uy) << OP_GT;
  cu |= (uint32_t) (ux >= uy) << OP_GE;

  assert(cb == ref); // C builtins
  assert(cp == ref); // packed operators
  assert(cu == ref); // unpacked operators
}

// Mostly arbitrary bit patterns, with one draw in sixteen taken from the special
// values so the sweep keeps hitting the classes an encoding rarely produces.
TEST_HOST_DEVICE_FUNC double draw(cuda::std::minstd_rand& rng, const double* specials, int n)
{
  cuda::std::uniform_int_distribution<int> one_in_16(0, 15);
  cuda::std::uniform_int_distribution<int> pick(0, n - 1);
  cuda::std::uniform_int_distribution<uint64_t> bits;

  return (one_in_16(rng) == 0) ? specials[pick(rng)] : cuda::std::bit_cast<double>(bits(rng));
}

// Exhaustively compares every ordered pair of the representative special values
// (covers the NaN / inf / zero / subnormal corners) across all three surfaces.
TEST_FUNC void test()
{
  const double specials[] = {
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    3.14159265358979,
    -3.14159265358979,
    1.0e308,
    -1.0e308,
    1.0e-308,
    -1.0e-308,
    from_bits(0x7FF0000000000000ULL), // +inf
    from_bits(0xFFF0000000000000ULL), // -inf
    from_bits(0x0010000000000000ULL), // min normal
    from_bits(0x8010000000000000ULL), // -min normal
    from_bits(0x000FFFFFFFFFFFFFULL), // max subnormal
    from_bits(0x0000000000000001ULL), // min subnormal
    from_bits(0x8000000000000001ULL), // -min subnormal
    from_bits(0x7FEFFFFFFFFFFFFFULL), // max finite
    from_bits(0xFFEFFFFFFFFFFFFFULL), // -max finite
    from_bits(0x7FF8000000000000ULL), // +qNaN
    from_bits(0xFFF8000000000000ULL), // -qNaN
    from_bits(0x7FF0000000000001ULL), // +sNaN
    from_bits(0xFFF0000000000001ULL), // -sNaN
    from_bits(0x7FF80000DEADBEEFULL), // qNaN with payload
  };
  const int n = (int) (sizeof(specials) / sizeof(specials[0]));

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      check_pair(specials[i], specials[j]);
    }
  }

  // Pseudo-random pairs. Every fourth pair compares a value against itself,
  // which is the only cheap way to reach the equal branch with random inputs.
  cuda::std::minstd_rand rng(0xC0FFEEu);
  cuda::std::uniform_int_distribution<int> one_in_4(0, 3);
  for (int i = 0; i < 512; i++)
  {
    const double x = draw(rng, specials, n);
    const double y = (one_in_4(rng) == 0) ? x : draw(rng, specials, n);
    check_pair(x, y);
  }
}

int main(int, char**)
{
  test();

  return 0;
}
