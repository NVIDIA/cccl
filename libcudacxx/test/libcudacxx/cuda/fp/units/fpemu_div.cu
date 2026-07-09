// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu division (correctly rounded, bit-exact).
//
//  Validates that the fpemu division reproduces, bit-for-bit, correctly-rounded
//  IEEE-754 binary64 division for all four rounding modes (rn, rz, ru, rd) across
//  the C builtins (__fp64emu_ddiv_*), the packed operator/ (rn) and the unpacked
//  operator/ (rn). The reference is the CUDA __ddiv_* intrinsics on the device and
//  fenv-directed division on the host, so the same _CCCL_HOST_DEVICE check runs on
//  the host and, under CUDA, on the device. NaN results are matched by class.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#if !defined(__CUDA_ARCH__)
#  include <cfenv>
#endif

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

enum
{
  M_RN = 0,
  M_RZ,
  M_RU,
  M_RD,
  M_COUNT
};

static double from_d_bits(uint64_t b)
{
  return ::cuda::std::bit_cast<double>(b);
}
_CCCL_HOST_DEVICE static uint64_t d_bits(double d)
{
  return ::cuda::std::bit_cast<uint64_t>(d);
}

_CCCL_HOST_DEVICE static bool is_nan_bits(uint64_t b)
{
  return ((b & UINT64_C(0x7FF0000000000000)) == UINT64_C(0x7FF0000000000000)) && (b & UINT64_C(0x000FFFFFFFFFFFFF));
}
// NaN payloads are platform-defined: treat any two NaNs as a match.
_CCCL_HOST_DEVICE static bool match(uint64_t got, uint64_t ref)
{
  return (got == ref) || (is_nan_bits(got) && is_nan_bits(ref));
}

// Reference: CUDA __ddiv_* intrinsics on device, fenv-directed division on host.
_CCCL_HOST_DEVICE static uint64_t ref_one(double a, double b, int mode)
{
#if defined(__CUDA_ARCH__)
  double q;
  switch (mode)
  {
    case M_RZ:
      q = __ddiv_rz(a, b);
      break;
    case M_RU:
      q = __ddiv_ru(a, b);
      break;
    case M_RD:
      q = __ddiv_rd(a, b);
      break;
    default:
      q = __ddiv_rn(a, b);
      break; // M_RN
  }
  return d_bits(q);
#else
  int old = fegetround();
  int fe;
  switch (mode)
  {
    case M_RZ:
      fe = FE_TOWARDZERO;
      break;
    case M_RU:
      fe = FE_UPWARD;
      break;
    case M_RD:
      fe = FE_DOWNWARD;
      break;
    default:
      fe = FE_TONEAREST;
      break; // M_RN
  }
  fesetround(fe);
  // volatile forces the divsd to execute (in memory) between the fesetround
  // calls and prevents compile-time constant folding under the wrong mode.
  volatile double va = a, vb = b;
  volatile double q = va / vb;
  double r          = q;
  fesetround(old);
  return d_bits(r);
#endif
}

// Compare every division surface for one pair against the reference on the same
// target. Returns true if all match.
_CCCL_HOST_DEVICE static bool check_pair(double x, double y)
{
  __fpbits64 a = __fp64emu_from_double(x);
  __fpbits64 b = __fp64emu_from_double(y);
  bool ok      = true;

  ok = ok && match((uint64_t) __fp64emu_ddiv_rn(a, b), ref_one(x, y, M_RN));
  ok = ok && match((uint64_t) __fp64emu_ddiv_rz(a, b), ref_one(x, y, M_RZ));
  ok = ok && match((uint64_t) __fp64emu_ddiv_ru(a, b), ref_one(x, y, M_RU));
  ok = ok && match((uint64_t) __fp64emu_ddiv_rd(a, b), ref_one(x, y, M_RD));

  fp64emu pa = x, pb = y;
  ok = ok && match(d_bits((double) (pa / pb)), ref_one(x, y, M_RN));

  fp64emu_unpacked ua = (fp64emu_unpacked) x, ub = (fp64emu_unpacked) y;
  ok = ok && match(d_bits((double) (ua / ub)), ref_one(x, y, M_RN));

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void kern_check(const double* x, const double* y, int n, int* mism)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (!check_pair(x[i], y[i]))
    {
      atomicAdd(mism, 1);
    }
  }
}

// Returns device mismatch count, or -1 on a CUDA API failure.
static int device_mismatches(const double* xs, const double* ys, int n)
{
  double* dx = nullptr;
  double* dy = nullptr;
  int* dm    = nullptr;
  if (cudaMallocManaged(&dx, n * sizeof(double)) != cudaSuccess)
  {
    return -1;
  }
  if (cudaMallocManaged(&dy, n * sizeof(double)) != cudaSuccess)
  {
    cudaFree(dx);
    return -1;
  }
  if (cudaMallocManaged(&dm, sizeof(int)) != cudaSuccess)
  {
    cudaFree(dx);
    cudaFree(dy);
    return -1;
  }
  ::memcpy(dx, xs, n * sizeof(double));
  ::memcpy(dy, ys, n * sizeof(double));
  *dm               = 0;
  const int threads = 256;
  int blocks        = (n + threads - 1) / threads;
  if (blocks > 1024)
  {
    blocks = 1024;
  }
  kern_check<<<blocks, threads>>>(dx, dy, n, dm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
  {
    err = cudaDeviceSynchronize();
  }
  const int m = (err == cudaSuccess) ? *dm : -1;
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dm);
  return m;
}
#endif // _CCCL_CUDA_COMPILATION()

// Host loop (+ device kernel under CUDA); returns total mismatches.
static int run_dataset(const char* label, const double* xs, const double* ys, int n)
{
  int mism = 0;
  for (int i = 0; i < n; i++)
  {
    if (!check_pair(xs[i], ys[i]))
    {
      ++mism;
    }
  }
#if _CCCL_CUDA_COMPILATION()
  const int dev = device_mismatches(xs, ys, n);
  mism += (dev < 0) ? 1 : dev;
#endif // _CCCL_CUDA_COMPILATION()
  ::printf("  %-16s: %7d pairs, %d mismatches\n", label, n, mism);
  return mism;
}

static const double g_special[] = {
  0.0,
  -0.0,
  1.0,
  -1.0,
  2.0,
  -2.0,
  3.0,
  -3.0,
  0.5,
  -0.5,
  100.0,
  -100.0,
  3.14159265358979,
  -3.14159265358979,
  1e-300,
  -1e-300,
  1e300,
  -1e300,
  from_d_bits(0x0000000000000001ULL), // min subnormal
  from_d_bits(0x800FFFFFFFFFFFFFULL), // -max subnormal
  from_d_bits(0x0010000000000000ULL), // min normal
  HUGE_VAL,
  -HUGE_VAL, // +inf, -inf
  from_d_bits(0x7FF8000000000000ULL), // +qNaN
  from_d_bits(0xFFF8000000000000ULL), // -qNaN
  from_d_bits(0x7FF0000000000001ULL), // +sNaN
};
static const int g_special_n = (int) (sizeof(g_special) / sizeof(g_special[0]));

static double rand_one(std::mt19937_64& gen)
{
  std::uniform_real_distribution<double> small(-4.0, 4.0);
  std::uniform_real_distribution<double> med(-1.0e150, 1.0e150);
  switch ((int) (gen() % 4))
  {
    case 0:
      return g_special[gen() % g_special_n];
    case 1:
      return small(gen);
    case 2:
      return med(gen);
    default:
      return from_d_bits(gen());
  }
}

C2H_TEST("fpemu division (correctly rounded, bit-exact)", "[fpemu]")
{
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  // All ordered pairs of the special values.
  std::vector<double> sx, sy;
  for (int i = 0; i < g_special_n; i++)
  {
    for (int j = 0; j < g_special_n; j++)
    {
      sx.push_back(g_special[i]);
      sy.push_back(g_special[j]);
    }
  }
  REQUIRE(run_dataset("special pairs", sx.data(), sy.data(), (int) sx.size()) == 0);

  // Random pairs spanning the full exponent range (incl. subnormals).
  constexpr int NR = 300000;
  std::vector<double> rx(NR), ry(NR);
  std::mt19937_64 gen(0xD1D1DEu);
  for (int i = 0; i < NR; i++)
  {
    rx[i] = rand_one(gen);
    ry[i] = rand_one(gen);
  }
  REQUIRE(run_dataset("random pairs", rx.data(), ry.data(), NR) == 0);
}
