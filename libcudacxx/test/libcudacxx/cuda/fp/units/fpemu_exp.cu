// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: exp() implemented on fp64emu vs std::exp.
//
//  A generic exp_impl<T>() (polynomial approximation + range reduction) is
//  instantiated for the emulated double fp64emu and its result is compared against
//  the native std::exp reference with a relative-error bound. The same generic
//  code path proves fp64emu can back an application-level transcendental. The
//  _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

constexpr double epsilon = 1e-4;

// exp() via range reduction + polynomial, generic over double and fp64emu.
template <typename T>
_CCCL_HOST_DEVICE T exp_impl(T x)
{
#define LN2_HI  0x1.62e42fefa39efp-1 // high part of ln(2)
#define LN2_LO  0x1.abc9e3b39803fp-34 // low part for extra precision
#define INV_LN2 0x1.71547652b82fep+0 // 1 / ln(2)

  if (x != x)
  {
    return x; // NaN
  }
  if (x > 709.782712893384)
  {
    return T(1.0) / 0.0; // overflow
  }
  if (x < -745.1332191019411)
  {
    return T(0.0); // underflow
  }

  // Range reduction: x = k * ln2 + r,  |r| <= ln2/2.
  int k = (int) (x * INV_LN2 + (x >= 0 ? 0.5 : -0.5));

#if defined(__CUDA_ARCH__)
  T r = __fma_rn(-k, LN2_HI, x);
  r   = __fma_rn(-k, LN2_LO, r);
#else
  T r = fma(-k, LN2_HI, x);
  r   = fma(-k, LN2_LO, r);
#endif

  // Polynomial approximation of exp(r), r in [-ln2/2, ln2/2].
  T poly =
    0x1p+0
    + r
        * (0x1p+0
           + r
               * (0x1p-1
                  + r
                      * (0x1.5555555555555p-3
                         + r * (0x1.999999999999ap-5 + r * (0x1.6c16c16c16c17p-7 + r * (0x1.a01a01a01a01ap-9))))));

  // Reconstruct exp(x) = 2^k * exp(r). Bias = 1023 for double.
  int exponent = k + 1023;
  if (exponent <= 0) // subnormal
  {
    if (exponent < -52)
    {
      return T(0.0);
    }
    uint64_t uexp = (uint64_t) (exponent + 52) << 52;
    T dexp        = ::cuda::std::bit_cast<T>(uexp);
    return poly * dexp * 0x1.0p-52;
  }

  if (exponent >= 2047)
  {
    return T(1.0) / 0.0;
  }

  uint64_t uexp = (uint64_t) exponent << 52;
  T dexp        = ::cuda::std::bit_cast<T>(uexp);
  return poly * dexp;

#undef LN2_HI
#undef LN2_LO
#undef INV_LN2
}

// Evaluate exp_impl<fp64emu> across a fixed set of inputs and verify each stays
// within the relative-error bound of std::exp.
_CCCL_HOST_DEVICE bool run_test()
{
  const double tv[10] = {0.0, 0.00001, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0, 700.0, -700.0};

  bool ok = true;
  for (int i = 0; i < 10; i++)
  {
    const double ref = ::cuda::std::exp(tv[i]);
    const double got = (double) exp_impl<fp64emu>(tv[i]);
    const double rel = (ref != 0.0) ? ::cuda::std::fabs(got - ref) / ref : 0.0;
    ok               = ok && (rel < epsilon);
  }
  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu exp() vs std::exp", "[fpemu]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
