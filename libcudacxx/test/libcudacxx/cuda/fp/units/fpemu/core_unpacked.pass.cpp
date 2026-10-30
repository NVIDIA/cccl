// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu packed + unpacked core builtins (mul/add/mad/dot/poly).
//
//  Drives the low-level packed (__fp64emu_*) and unpacked (__fp64emu_unpacked_*)
//  emulation cores on raw __fpbits64 / __fpbits64_unpacked values for a small set
//  of composite operations, checking each against the native double reference
//  within a tight tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cmath>

#include "test_macros.h"

#define C0 (1.0)
#define C1 (1.0 / 2.0)
#define C2 (1.0 / 6.0)
#define C3 (1.0 / 24.0)
#define C4 (1.0 / 120.0)
#define C5 (1.0 / 720.0)
#define C6 (1.0 / 5040.0)
#define C7 (1.0 / 40320.0)

TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz, double dw)
{
  const double ref[5] = {
    dx * dy * dz * dw,
    dx + dy + dz + dw,
    dx * dy + dz,
    dx * dy + dz * dw,
    C0 + dx * (C1 + dx * (C2 + dx * (C3 + dx * (C4 + dx * (C5 + dx * (C6 + dx * C7)))))),
  };

  // Packed cores on __fpbits64.
  cuda::__fpbits64 ex = cuda::__fp64emu_from_double(dx);
  cuda::__fpbits64 ey = cuda::__fp64emu_from_double(dy);
  cuda::__fpbits64 ez = cuda::__fp64emu_from_double(dz);
  cuda::__fpbits64 ew = cuda::__fp64emu_from_double(dw);

  cuda::__fpbits64 pmul =
    cuda::__fp64emu_mid_dmul_rn(cuda::__fp64emu_mid_dmul_rn(cuda::__fp64emu_mid_dmul_rn(ex, ey), ez), ew);
  cuda::__fpbits64 padd =
    cuda::__fp64emu_mid_dadd_rn(cuda::__fp64emu_mid_dadd_rn(cuda::__fp64emu_mid_dadd_rn(ex, ey), ez), ew);
  cuda::__fpbits64 pmad  = cuda::__fp64emu_mid_mad_rn(ex, ey, ez);
  cuda::__fpbits64 pdot  = cuda::__fp64emu_mid_dot_rn(ex, ez, ey, ew);
  cuda::__fpbits64 ppoly = cuda::__fp64emu_dmul_rn(ex, cuda::__fp64emu_from_double(C7));
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C6)), ex);
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C5)), ex);
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C4)), ex);
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C3)), ex);
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C2)), ex);
  ppoly                  = cuda::__fp64emu_dmul_rn(cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C1)), ex);
  ppoly                  = cuda::__fp64emu_dadd_rn(ppoly, cuda::__fp64emu_from_double(C0));
  const double packed[5] = {
    cuda::__fp64emu_to_double(pmul),
    cuda::__fp64emu_to_double(padd),
    cuda::__fp64emu_to_double(pmad),
    cuda::__fp64emu_to_double(pdot),
    cuda::__fp64emu_to_double(ppoly),
  };

  // Unpacked cores on __fpbits64_unpacked.
  cuda::__fpbits64_unpacked ux = cuda::__fp64emu_unpacked_from_double(dx);
  cuda::__fpbits64_unpacked uy = cuda::__fp64emu_unpacked_from_double(dy);
  cuda::__fpbits64_unpacked uz = cuda::__fp64emu_unpacked_from_double(dz);
  cuda::__fpbits64_unpacked uw = cuda::__fp64emu_unpacked_from_double(dw);

  cuda::__fpbits64_unpacked umul = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dmul(cuda::__fp64emu_unpacked_mid_dmul(ux, uy), uz), uw);
  cuda::__fpbits64_unpacked uadd = cuda::__fp64emu_unpacked_mid_dadd(
    cuda::__fp64emu_unpacked_mid_dadd(cuda::__fp64emu_unpacked_mid_dadd(ux, uy), uz), uw);
  cuda::__fpbits64_unpacked umad  = __fp64emu_unpacked_mid_mad(ux, uy, uz);
  cuda::__fpbits64_unpacked udot  = __fp64emu_unpacked_mid_dot(ux, uz, uy, uw);
  cuda::__fpbits64_unpacked upoly = cuda::__fp64emu_unpacked_mid_dmul(ux, cuda::__fp64emu_unpacked_from_double(C7));
  upoly                           = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C6)), ux);
  upoly = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C5)), ux);
  upoly = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C4)), ux);
  upoly = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C3)), ux);
  upoly = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C2)), ux);
  upoly = cuda::__fp64emu_unpacked_mid_dmul(
    cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C1)), ux);
  upoly                    = cuda::__fp64emu_unpacked_mid_dadd(upoly, cuda::__fp64emu_unpacked_from_double(C0));
  const double unpacked[5] = {
    cuda::__fp64emu_unpacked_to_double(umul),
    cuda::__fp64emu_unpacked_to_double(uadd),
    cuda::__fp64emu_unpacked_to_double(umad),
    cuda::__fp64emu_unpacked_to_double(udot),
    cuda::__fp64emu_unpacked_to_double(upoly),
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
