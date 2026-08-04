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
//  and the unpacked operators. The same _CCCL_HOST_DEVICE check runs on the host
//  and, under CUDA, on the device.
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

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

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

static double from_d_bits(uint64_t b)
{
  return ::cuda::std::bit_cast<double>(b);
}

// Pack the six native comparison results into one bit code.
_CCCL_HOST_DEVICE static uint32_t native_codes(double x, double y)
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
_CCCL_HOST_DEVICE static bool check_pair(double x, double y)
{
  const uint32_t ref = native_codes(x, y);

  __fpbits64 ex = __fp64emu_from_double(x);
  __fpbits64 ey = __fp64emu_from_double(y);
  uint32_t cb   = 0;
  cb |= (uint32_t) __fp64emu_cmp_eq(ex, ey) << OP_EQ;
  cb |= (uint32_t) __fp64emu_cmp_ne(ex, ey) << OP_NE;
  cb |= (uint32_t) __fp64emu_cmp_lt(ex, ey) << OP_LT;
  cb |= (uint32_t) __fp64emu_cmp_le(ex, ey) << OP_LE;
  cb |= (uint32_t) __fp64emu_cmp_gt(ex, ey) << OP_GT;
  cb |= (uint32_t) __fp64emu_cmp_ge(ex, ey) << OP_GE;

  fp64emu px = x, py = y;
  uint32_t cp = 0;
  cp |= (uint32_t) (px == py) << OP_EQ;
  cp |= (uint32_t) (px != py) << OP_NE;
  cp |= (uint32_t) (px < py) << OP_LT;
  cp |= (uint32_t) (px <= py) << OP_LE;
  cp |= (uint32_t) (px > py) << OP_GT;
  cp |= (uint32_t) (px >= py) << OP_GE;

  fp64emu_unpacked ux = (fp64emu_unpacked) x, uy = (fp64emu_unpacked) y;
  uint32_t cu = 0;
  cu |= (uint32_t) (ux == uy) << OP_EQ;
  cu |= (uint32_t) (ux != uy) << OP_NE;
  cu |= (uint32_t) (ux < uy) << OP_LT;
  cu |= (uint32_t) (ux <= uy) << OP_LE;
  cu |= (uint32_t) (ux > uy) << OP_GT;
  cu |= (uint32_t) (ux >= uy) << OP_GE;

  return cb == ref && cp == ref && cu == ref;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void kern_check(const double* x, const double* y, int n, int* mism)
{
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += static_cast<int>(blockDim.x * gridDim.x))
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
  ::printf("  %-16s: %6d pairs, %d mismatches\n", label, n, mism);
  return mism;
}

static const double g_special[] = {
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
  HUGE_VAL,
  -HUGE_VAL, // +inf, -inf
  from_d_bits(0x0010000000000000ULL), // min normal
  from_d_bits(0x8010000000000000ULL), // -min normal
  from_d_bits(0x000FFFFFFFFFFFFFULL), // max subnormal
  from_d_bits(0x0000000000000001ULL), // min subnormal
  from_d_bits(0x8000000000000001ULL), // -min subnormal
  from_d_bits(0x7FEFFFFFFFFFFFFFULL), // max finite
  from_d_bits(0xFFEFFFFFFFFFFFFFULL), // -max finite
  from_d_bits(0x7FF8000000000000ULL), // +qNaN
  from_d_bits(0xFFF8000000000000ULL), // -qNaN
  from_d_bits(0x7FF0000000000001ULL), // +sNaN
  from_d_bits(0xFFF0000000000001ULL), // -sNaN
  from_d_bits(0x7FF80000DEADBEEFULL), // qNaN with payload
};
static const int g_special_n = (int) (sizeof(g_special) / sizeof(g_special[0]));

C2H_TEST("fpemu comparison vs native double", "[fpemu]")
{
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  // All ordered pairs of the special values (covers NaN/inf/zero corners).
  std::vector<double> sx, sy;
  for (const double vi : g_special)
  {
    for (const double vj : g_special)
    {
      sx.push_back(vi);
      sy.push_back(vj);
    }
  }
  REQUIRE(run_dataset("special pairs", sx.data(), sy.data(), (int) sx.size()) == 0);

  // Random finite + occasional NaN/inf pairs, with frequent magnitude ties.
  constexpr int N = 65536;
  std::vector<double> rx(N), ry(N);
  std::mt19937_64 gen(0xC0FFEE);
  auto gen_one = [&]() -> double {
    uint64_t r = gen();
    if (((r >> 60) & 0xF) == 0)
    {
      return g_special[gen() % g_special_n];
    }
    return from_d_bits(r);
  };
  for (int i = 0; i < N; i++)
  {
    rx[i] = gen_one();
    ry[i] = ((gen() & 3) == 0) ? rx[i] : gen_one();
  }
  REQUIRE(run_dataset("random pairs", rx.data(), ry.data(), N) == 0);
}
