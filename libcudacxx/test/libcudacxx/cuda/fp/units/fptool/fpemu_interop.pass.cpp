// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: <cuda/fpemu> and <cuda/fptool> used in one translation unit.
//
//  Both libraries live in namespace cuda::experimental, and fpemu declares
//  overloads named after the CUDA rounding intrinsics (__dadd_rn, __dsub_rn,
//  __dmul_rn, __ddiv_rn, __dsqrt_rn, __fma_rn) for its own emulated types.
//  fp_custom's device paths call the global intrinsics of the same name, so an
//  unqualified call from inside cuda::experimental finds fpemu's overload set,
//  stops there, and never reaches the global scope: every such call fails to
//  compile as soon as both headers are included. fp_custom therefore spells them
//  ::__dadd_rn.
//
//  This is primarily a compile-time guard, which is why fpemu is included first
//  (the order that puts its declarations in scope before fp_custom's bodies) and
//  why the arithmetic runs in a TEST_HOST_DEVICE_FUNC: the shadowed calls exist only in the
//  device paths, so they have to be instantiated to be checked. The runtime
//  assertions catch the quieter failure mode where a wrong-but-viable overload is
//  selected and the arithmetic silently changes.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

// Include order is load-bearing here; see above.
#include <cuda/fpemu>
#include <cuda/fptool>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Every fp_custom operation whose device path calls one of the captured names.
// At the native field sizes fp64_custom is a drop-in for double, so the
// expected values are exact.
TEST_HOST_DEVICE_FUNC bool check_fptool()
{
  const cudax::fp64_custom<> a(3.0);
  const cudax::fp64_custom<> b(4.0);
  const cudax::fp64_custom<> c(0.5);

  bool ok = true;

  ok = ok && static_cast<double>(a + b) == 7.0; // __dadd_rn
  ok = ok && static_cast<double>(a - b) == -1.0; // __dsub_rn
  ok = ok && static_cast<double>(a * b) == 12.0; // __dmul_rn
  ok = ok && static_cast<double>(b / cudax::fp64_custom<>(2.0)) == 2.0; // __ddiv_rn
  ok = ok && static_cast<double>(sqrt(cudax::fp64_custom<>(16.0))) == 4.0; // __dsqrt_rn
  ok = ok && static_cast<double>(fma(a, b, c)) == 12.5; // __fma_rn

  // The compound and inc/dec operators reach the same intrinsics through the
  // binary operators.
  {
    cudax::fp64_custom<> x(1.0);
    x += a;
    x -= c;
    x *= b;
    x /= cudax::fp64_custom<>(2.0);
    ++x;
    --x;
    ok = ok && static_cast<double>(x) == 7.0;
  }

  return ok;
}

// The other direction: fpemu's intrinsic-named overloads must still be found for
// fpemu's own types. Only add and sub are exercised, because nvc++ 26.3 cannot
// compile fpemu's mul/fma/div paths at all -- it hits "Unhandled builtin
// function" on the __umul64hi inside fpemu's __mul_128 and inside cuda::mul_hi.
TEST_HOST_DEVICE_FUNC bool check_fpemu()
{
  const double dx = 1.2345;
  const double dy = 2.3456;

  const cudax::fp64emu ex = dx;
  const cudax::fp64emu ey = dy;

  // Unqualified, so these resolve to cuda::experimental::__dadd_rn and friends.
  static_assert(::cuda::std::is_same_v<decltype(cudax::__dadd_rn(ex, ey)), cudax::fp64emu>);
  static_assert(::cuda::std::is_same_v<decltype(cudax::__dsub_rn(ex, ey)), cudax::fp64emu>);

  const double tol = 1e-10;

  bool ok = true;
  ok      = ok && ::cuda::std::fabs(static_cast<double>(cudax::__dadd_rn(ex, ey)) - (dx + dy)) <= tol;
  ok      = ok && ::cuda::std::fabs(static_cast<double>(cudax::__dsub_rn(ex, ey)) - (dx - dy)) <= tol;

  return ok;
}

// Values crossing between the two libraries, so both are live in one expression.
TEST_HOST_DEVICE_FUNC bool check_mixed()
{
  const double dx = 6.25;
  const double dy = 1.5;

  const cudax::fp64emu emu_sum        = cudax::fp64emu(dx) + cudax::fp64emu(dy);
  const cudax::fp64_custom<> tool_sum = cudax::fp64_custom<>(dx) + cudax::fp64_custom<>(dy);

  bool ok = static_cast<double>(tool_sum) == dx + dy;
  ok      = ok && static_cast<double>(emu_sum) == static_cast<double>(tool_sum);

  // Feed an fpemu result into fp_custom and back.
  const cudax::fp64_custom<> round_trip =
    cudax::fp64_custom<>(static_cast<double>(emu_sum)) * cudax::fp64_custom<>(2.0);
  ok = ok && static_cast<double>(round_trip) == 2.0 * (dx + dy);

  return ok;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(check_fptool());
  assert(check_fpemu());
  assert(check_mixed());
}

int main(int, char**)
{
  test();

  return 0;
}
