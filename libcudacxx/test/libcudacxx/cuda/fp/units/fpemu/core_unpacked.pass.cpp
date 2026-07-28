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

#include <cuda/fpemu>
#include <cuda/std/cmath>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#define C0 (1.0)
#define C1 (1.0 / 2.0)
#define C2 (1.0 / 6.0)
#define C3 (1.0 / 24.0)
#define C4 (1.0 / 120.0)
#define C5 (1.0 / 720.0)
#define C6 (1.0 / 5040.0)
#define C7 (1.0 / 40320.0)

TEST_FUNC void test(double dx, double dy, double dz, double dw)
{
  const double ref[5] = {
    dx * dy * dz * dw,
    dx + dy + dz + dw,
    dx * dy + dz,
    dx * dy + dz * dw,
    C0 + dx * (C1 + dx * (C2 + dx * (C3 + dx * (C4 + dx * (C5 + dx * (C6 + dx * C7)))))),
  };

  // Packed cores on __fpbits64.
  __fpbits64 ex = __fp64emu_from_double(dx);
  __fpbits64 ey = __fp64emu_from_double(dy);
  __fpbits64 ez = __fp64emu_from_double(dz);
  __fpbits64 ew = __fp64emu_from_double(dw);

  __fpbits64 pmul        = __fp64emu_mid_dmul_rn(__fp64emu_mid_dmul_rn(__fp64emu_mid_dmul_rn(ex, ey), ez), ew);
  __fpbits64 padd        = __fp64emu_mid_dadd_rn(__fp64emu_mid_dadd_rn(__fp64emu_mid_dadd_rn(ex, ey), ez), ew);
  __fpbits64 pmad        = __fp64emu_mid_mad_rn(ex, ey, ez);
  __fpbits64 pdot        = __fp64emu_mid_dot_rn(ex, ez, ey, ew);
  __fpbits64 ppoly       = __fp64emu_dmul_rn(ex, __fp64emu_from_double(C7));
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C6)), ex);
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C5)), ex);
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C4)), ex);
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C3)), ex);
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C2)), ex);
  ppoly                  = __fp64emu_dmul_rn(__fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C1)), ex);
  ppoly                  = __fp64emu_dadd_rn(ppoly, __fp64emu_from_double(C0));
  const double packed[5] = {
    __fp64emu_to_double(pmul),
    __fp64emu_to_double(padd),
    __fp64emu_to_double(pmad),
    __fp64emu_to_double(pdot),
    __fp64emu_to_double(ppoly),
  };

  // Unpacked cores on __fpbits64_unpacked.
  __fpbits64_unpacked ux = __fp64emu_unpacked_from_double(dx);
  __fpbits64_unpacked uy = __fp64emu_unpacked_from_double(dy);
  __fpbits64_unpacked uz = __fp64emu_unpacked_from_double(dz);
  __fpbits64_unpacked uw = __fp64emu_unpacked_from_double(dw);

  __fpbits64_unpacked umul =
    __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dmul(ux, uy), uz), uw);
  __fpbits64_unpacked uadd =
    __fp64emu_unpacked_mid_dadd(__fp64emu_unpacked_mid_dadd(__fp64emu_unpacked_mid_dadd(ux, uy), uz), uw);
  __fpbits64_unpacked umad  = __fp64emu_unpacked_mid_mad(ux, uy, uz);
  __fpbits64_unpacked udot  = __fp64emu_unpacked_mid_dot(ux, uz, uy, uw);
  __fpbits64_unpacked upoly = __fp64emu_unpacked_mid_dmul(ux, __fp64emu_unpacked_from_double(C7));
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C6)), ux);
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C5)), ux);
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C4)), ux);
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C3)), ux);
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C2)), ux);
  upoly = __fp64emu_unpacked_mid_dmul(__fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C1)), ux);
  upoly = __fp64emu_unpacked_mid_dadd(upoly, __fp64emu_unpacked_from_double(C0));
  const double unpacked[5] = {
    __fp64emu_unpacked_to_double(umul),
    __fp64emu_unpacked_to_double(uadd),
    __fp64emu_unpacked_to_double(umad),
    __fp64emu_unpacked_to_double(udot),
    __fp64emu_unpacked_to_double(upoly),
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
