// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: <cuda/fpemu> and <cuda/fpmp> used in one translation unit.
//
//  Both libraries live in namespace cuda::experimental, and fpemu declares
//  overloads named after the CUDA rounding intrinsics (__dadd_rn, __dmul_rn,
//  __double2int_rz, ...) for its own emulated types. fpmp's device paths call the
//  global intrinsics of the same name, so an unqualified call from inside
//  cuda::experimental finds fpemu's overload set, stops there, and never reaches
//  the global scope -- every such call fails to compile the moment both headers
//  are included. fpmp therefore spells them ::__dadd_rn.
//
//  This test is primarily a compile-time guard, which is why fpemu is included
//  first (the order that puts its declarations in scope before fpmp's bodies) and
//  why the fpmp arithmetic runs in a TEST_HOST_DEVICE_FUNC: the shadowed calls only exist in
//  the device paths, so they have to be instantiated to be checked. The runtime
//  assertions catch the quieter failure mode where a wrong-but-viable overload is
//  selected and the arithmetic silently changes.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

// Include order is load-bearing here; see above.
#include <cuda/fpemu>
#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// fpmp arithmetic and conversions. The calls fpemu can capture are the fp64 ones
// (__dadd_rn, __dadd_rz, __dsub_rn, __dmul_rn) and the __double2* converters, since
// those are the names it declares; the fp32 paths are covered too because fpmp
// qualifies every intrinsic uniformly.
template <class _Tp>
TEST_HOST_DEVICE_FUNC void check_fpmp()
{
  using T = _Tp;

  const T a(3.0);
  const T b(4.0);
  const T c(0.5);

  assert(static_cast<double>(a + b) == 7.0);
  assert(static_cast<double>(a - b) == -1.0);
  assert(static_cast<double>(a * b) == 12.0);
  assert(static_cast<double>(b / T(2.0)) == 2.0);
  assert(static_cast<double>(fma(a, b, c)) == 12.5);
  assert(static_cast<double>(sqrt(T(16.0))) == 4.0);

  // Round-off sensitive: 2^-30 added to 1 survives in the low word, which native
  // single precision could not hold. A wrong add would collapse this to 1.
  if constexpr (::cuda::std::is_same_v<T, cudax::fp32mp2>)
  {
    const T eps(1.0 / 1073741824.0); // 2^-30
    assert(static_cast<double>((T(1.0) + eps) - T(1.0)) == 1.0 / 1073741824.0);
  }

  // Float-to-integer conversions: the __double2int_rz / __float2int_rz family.
  assert(static_cast<int>(T(-2.75)) == -2);
  assert(static_cast<int>(T(7.9)) == 7);
  assert(static_cast<unsigned>(T(7.9)) == 7u);
  assert(static_cast<long long>(T(-7.9)) == -7ll);
  assert(static_cast<unsigned long long>(T(7.9)) == 7ull);

  // Integer-to-float conversions: the __int2float_rn / __ll2double_rz family.
  assert(static_cast<double>(T(-9)) == -9.0);
  assert(static_cast<double>(T(123456789ll)) == 123456789.0);
}

// The other direction: fpemu's intrinsic-named overloads must still be found for
// fpemu's own types, exactly as fpemu/api.pass.cpp calls them.
TEST_HOST_DEVICE_FUNC void check_fpemu()
{
  const double dx = 1.2345;
  const double dy = 2.3456;
  const double dz = 3.4567;

  cudax::fp64emu ex = dx;
  cudax::fp64emu ey = dy;
  cudax::fp64emu ez = dz;

  // Unqualified, so these resolve to cuda::experimental::__dadd_rn and friends.
  // Division is left out because __ddiv_rn is not one of the names that can capture
  // an fpmp call. Note that nvc++ 26.3 cannot compile these fpemu paths at all: it
  // hits "Unhandled builtin function" on the __umul64hi in cuda::mul_hi and in
  // fpemu's __mul_128, which reaches fma and div alike.
  static_assert(::cuda::std::is_same_v<decltype(cudax::__dadd_rn(ex, ey)), cudax::fp64emu>);
  static_assert(::cuda::std::is_same_v<decltype(cudax::__dmul_rn(ex, ey)), cudax::fp64emu>);
  static_assert(::cuda::std::is_same_v<decltype(cudax::__fma_rn(ex, ey, ez)), cudax::fp64emu>);

  const double tol = 1e-10;

  assert(::cuda::std::fabs(static_cast<double>(cudax::__dadd_rn(ex, ey)) - (dx + dy)) <= tol);
  assert(::cuda::std::fabs(static_cast<double>(cudax::__dsub_rn(ex, ey)) - (dx - dy)) <= tol);
  assert(::cuda::std::fabs(static_cast<double>(cudax::__dmul_rn(ex, ey)) - (dx * dy)) <= tol);
  assert(::cuda::std::fabs(static_cast<double>(cudax::__fma_rn(ex, ey, ez)) - (dx * dy + dz)) <= tol);
}

// Values crossing between the two libraries, so both are live in one expression.
TEST_HOST_DEVICE_FUNC void check_mixed()
{
  const double dx = 6.25;
  const double dy = 1.5;

  const cudax::fp64emu emu_sum = cudax::fp64emu(dx) + cudax::fp64emu(dy);
  const cudax::fp64mp2 mp_sum  = cudax::fp64mp2(dx) + cudax::fp64mp2(dy);

  assert(static_cast<double>(emu_sum) == dx + dy);
  assert(static_cast<double>(mp_sum) == dx + dy);
  assert(static_cast<double>(mp_sum) == static_cast<double>(emu_sum));

  // Feed an fpemu result into fpmp and back.
  const cudax::fp64mp2 round_trip = cudax::fp64mp2(static_cast<double>(emu_sum)) * cudax::fp64mp2(2.0);
  assert(static_cast<double>(round_trip) == 2.0 * (dx + dy));
}

TEST_HOST_DEVICE_FUNC void test()
{
  check_fpmp<cudax::fp32mp2>();
  check_fpmp<cudax::fp64mp2>();
  check_fpemu();
  check_mixed();
}

int main(int, char**)
{
  test();

  return 0;
}
