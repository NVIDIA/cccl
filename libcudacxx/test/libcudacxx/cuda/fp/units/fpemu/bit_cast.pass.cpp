// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: bit_cast on emulated double values (packed fpemu and unpacked).
//
//  Verifies that:
//    - the packed fpemu<double> round-trips through its 64-bit IEEE-754
//      representation via cuda::std::bit_cast and is bit-identical to the
//      native double (bits is private; bit_cast is the supported reinterpret),
//    - the unpacked emulated double is layout-compatible with and trivially
//      copyable to its raw {sign, exponent, mantissa} representation, so an
//      equal-size cuda::std::bit_cast round-trips values exactly (there is no
//      size-changing bit_cast overload), produces the expected result of a simple
//      arithmetic expression, and yields an identical raw representation for a
//      plain conversion across all accuracy levels.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Same-size (16-byte) mirror of the unpacked representation {sign, exponent, mantissa}.
// fpemu_unpacked keeps its storage private and intentionally offers no size-changing
// bit_cast overload, so raw access goes through an equal-size cuda::std::bit_cast.
struct fpemu_unpacked_bits
{
  uint32_t sign;
  uint32_t exponent;
  uint64_t mantissa;
};

TEST_HOST_DEVICE_FUNC void test()
{
  // Packed fpemu<double>: reinterpret to/from the 64-bit IEEE-754 representation
  // via the standard bit_cast (fpemu::bits is private, so this is the only way).
  const double packed_vals[6] = {1.5, -2.0, 0.0, -0.0, 42.0, 3.14159265358979323846};
  for (const double packed_val : packed_vals)
  {
    const cudax::fpemu<double> p(packed_val);
    const uint64_t pbits = cuda::std::bit_cast<uint64_t>(p);
    // fpemu<double> is a faithful double, so its bits match the native double's.
    assert(pbits == cuda::std::bit_cast<uint64_t>(packed_val));
    // uint64_t -> fpemu<double> -> double round-trips the value exactly.
    assert(static_cast<double>(cuda::std::bit_cast<cudax::fpemu<double>>(pbits)) == packed_val);
  }

  // Unpacked fpemu is layout-compatible with, and trivially copyable to, its raw
  // {sign, exponent, mantissa} representation, so an equal-size bit_cast is the
  // supported way to reach the storage (there is no size-changing overload).
  static_assert(sizeof(cudax::fp64emu_unpacked) == sizeof(fpemu_unpacked_bits),
                "unpacked fpemu must be bit-compatible with its representation");
  static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64emu_unpacked>,
                "unpacked fpemu must be trivially copyable for bit_cast");

  // Round-trip: double -> unpacked -> (equal-size) bits -> unpacked -> value.
  const double test_vals[5] = {1.5, -2.0, 0.0, 42.0, 3.14159265358979323846};
  for (const double test_val : test_vals)
  {
    cudax::fp64emu_unpacked x(test_val);
    const auto rep = cuda::std::bit_cast<fpemu_unpacked_bits>(x);
    const auto y   = cuda::std::bit_cast<cudax::fp64emu_unpacked>(rep);
    assert(static_cast<double>(y) == test_val);
  }

  // Arithmetic result via value conversion: 2 * 3 + 1 == 7.
  cudax::fp64emu_unpacked a(2.0), b(3.0), c(1.0);
  assert(cuda::std::fabs(static_cast<double>(a * b + c) - 7.0) <= 1e-10);

  // A plain conversion produces an identical raw representation across accuracy levels.
  const double pi     = 3.14159265358979323846;
  const auto rep_def  = cuda::std::bit_cast<fpemu_unpacked_bits>(cudax::fp64emu_unpacked(pi));
  const auto rep_high = cuda::std::bit_cast<fpemu_unpacked_bits>(cudax::fp64emu_unpacked_high(pi));
  const auto rep_low  = cuda::std::bit_cast<fpemu_unpacked_bits>(cudax::fp64emu_unpacked_low(pi));
  assert(rep_def.sign == rep_high.sign);
  assert(rep_def.exponent == rep_high.exponent);
  assert(rep_def.mantissa == rep_high.mantissa);
  assert(rep_def.sign == rep_low.sign);
  assert(rep_def.exponent == rep_low.exponent);
  assert(rep_def.mantissa == rep_low.mantissa);
}

int main(int, char**)
{
  test();

  return 0;
}
