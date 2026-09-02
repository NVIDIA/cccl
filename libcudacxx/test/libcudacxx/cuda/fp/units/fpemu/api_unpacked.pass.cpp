// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu packed + unpacked C++ API (mul/add/mad/dot/poly).
//
//  Exercises the emulated double type through both the packed (fp64emu) and the
//  unpacked (fp64emu_unpacked) C++ API surfaces for a small set of composite
//  operations, and checks each against the native double reference within a tight
//  tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#define C0 (1.0)
#define C1 (1.0 / 2.0)
#define C2 (1.0 / 6.0)
#define C3 (1.0 / 24.0)
#define C4 (1.0 / 120.0)
#define C5 (1.0 / 720.0)
#define C6 (1.0 / 5040.0)
#define C7 (1.0 / 40320.0)

// Horner evaluation of the degree-7 polynomial for any value type.
#define POLY(v) (C0 + (v) * (C1 + (v) * (C2 + (v) * (C3 + (v) * (C4 + (v) * (C5 + (v) * (C6 + (v) * C7)))))))

TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz, double dw)
{
  const double ref[5] = {
    dx * dy * dz * dw,
    dx + dy + dz + dw,
    dx * dy + dz,
    dx * dy + dz * dw,
    POLY(dx),
  };

  // Packed C++ API.
  cudax::fp64emu ex = dx, ey = dy, ez = dz, ew = dw;
  const double packed[5] = {
    (double) (cudax::__dmul_rn(ex, ey) * ez * ew),
    (double) (cudax::__dadd_rn(ex, ey) + ez + ew),
    (double) cudax::mad(ex, ey, ez),
    (double) cudax::dot(ex, ez, ey, ew),
    (double) (POLY(ex)),
  };

  // Unpacked C++ API (explicit conversion to disambiguate from the packed type).
  cudax::fp64emu_unpacked ux = (cudax::fp64emu_unpacked) dx;
  cudax::fp64emu_unpacked uy = (cudax::fp64emu_unpacked) dy;
  cudax::fp64emu_unpacked uz = (cudax::fp64emu_unpacked) dz;
  cudax::fp64emu_unpacked uw = (cudax::fp64emu_unpacked) dw;
  const double unpacked[5]   = {
    (double) (cudax::__dmul_rn(ex, ey) * ez * ew),
    (double) (cudax::__dadd_rn(ux, uy) + uz + uw),
    (double) cudax::mad(ux, uy, uz),
    (double) cudax::dot(ux, uz, uy, uw),
    (double) (POLY(ux)),
  };

  const double tol = 1e-10;
  for (int i = 0; i < 5; i++)
  {
    assert(cuda::std::fabs(packed[i] - ref[i]) <= tol);
    assert(cuda::std::fabs(unpacked[i] - ref[i]) <= tol);
  }
}

int main(int, char**)
{
  const double dx = 0.23451432345642;
  const double dy = -2.34561234567899;
  const double dz = 3.45678726352678;
  const double dw = -4.56787263526789;
  test(dx, dy, dz, dw);

  return 0;
}
