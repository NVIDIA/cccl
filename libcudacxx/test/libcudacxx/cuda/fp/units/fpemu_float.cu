// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: float <-> double emulated conversions (fp64emu <-> float).
//
//  Validates that fpemu float->double (widening, exact) and double->float
//  (narrowing, round-to-nearest-even) conversions produce bit-identical results
//  to native casts across all value classes: normals, subnormals, +/-0, Inf, NaN
//  and rounding-boundary cases.
//
//  The per-value conversion + bit-exact check (f2d_ok / d2f_ok) is _CCCL_HOST_DEVICE
//  and is exercised on the host over host-generated datasets, and -- under CUDA --
//  over the same data on the device via a mismatch-counting kernel.
//
//  NaN payloads: when the packed API is routed through the unpacked cores
//  (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1) the unpack/pack round-trip
//  canonicalizes NaNs, so a NaN-vs-NaN bit difference is tolerated in that mode
//  only; otherwise the conversion must be strictly bit-exact.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>

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

// ---------------------------------------------------------------------------
// Bit-reinterpret helpers (host + device via cuda::std::bit_cast).
// ---------------------------------------------------------------------------
_CCCL_HOST_DEVICE inline uint64_t d_bits(double d)
{
  return ::cuda::std::bit_cast<uint64_t>(d);
}
_CCCL_HOST_DEVICE inline uint32_t f_bits(float f)
{
  return ::cuda::std::bit_cast<uint32_t>(f);
}
_CCCL_HOST_DEVICE inline double from_d_bits(uint64_t b)
{
  return ::cuda::std::bit_cast<double>(b);
}
_CCCL_HOST_DEVICE inline float from_f_bits(uint32_t b)
{
  return ::cuda::std::bit_cast<float>(b);
}

_CCCL_HOST_DEVICE inline bool is_nan_d(uint64_t b)
{
  return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFull) != 0;
}
_CCCL_HOST_DEVICE inline bool is_nan_f(uint32_t b)
{
  return ((b >> 23) & 0xFF) == 0xFF && (b & 0x007FFFFFu) != 0;
}

#if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
_CCCL_HOST_DEVICE constexpr bool relax_nan_payload()
{
  return true;
}
#else
_CCCL_HOST_DEVICE constexpr bool relax_nan_payload()
{
  return false;
}
#endif

// ---------------------------------------------------------------------------
// Per-value conversion + bit-exact check (host + device).
// ---------------------------------------------------------------------------

// float -> fp64emu -> double (widening, exact).
_CCCL_HOST_DEVICE inline bool f2d_ok(float v)
{
  fp64emu e(v);
  const uint64_t be = d_bits((double) e);
  const uint64_t br = d_bits((double) v);
  if (be == br)
  {
    return true;
  }
  return relax_nan_payload() && is_nan_d(be) && is_nan_d(br);
}

// double -> fp64emu -> float (narrowing, round-to-nearest-even).
_CCCL_HOST_DEVICE inline bool d2f_ok(double v)
{
  fp64emu e(v);
  const uint32_t be = f_bits((float) e);
  const uint32_t br = f_bits((float) v);
  if (be == br)
  {
    return true;
  }
  return relax_nan_payload() && is_nan_f(be) && is_nan_f(br);
}

// ---------------------------------------------------------------------------
// Device drivers: count mismatches over an array on the GPU.
// ---------------------------------------------------------------------------
#if _CCCL_CUDA_COMPILATION()
__global__ void kern_f2d(const float* v, int n, int* mism)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (!f2d_ok(v[i]))
    {
      atomicAdd(mism, 1);
    }
  }
}

__global__ void kern_d2f(const double* v, int n, int* mism)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (!d2f_ok(v[i]))
    {
      atomicAdd(mism, 1);
    }
  }
}

// Returns device mismatch count, or -1 on a CUDA API failure.
template <class T, class Kern>
static int device_mismatches(Kern kern, const T* vals, int n)
{
  T* dv    = nullptr;
  int* dm  = nullptr;
  if (cudaMallocManaged(&dv, n * sizeof(T)) != cudaSuccess)
  {
    return -1;
  }
  if (cudaMallocManaged(&dm, sizeof(int)) != cudaSuccess)
  {
    cudaFree(dv);
    return -1;
  }
  ::memcpy(dv, vals, n * sizeof(T));
  *dm               = 0;
  const int threads = 256;
  int blocks        = (n + threads - 1) / threads;
  if (blocks > 1024)
  {
    blocks = 1024;
  }
  kern<<<blocks, threads>>>(dv, n, dm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
  {
    err = cudaDeviceSynchronize();
  }
  const int m = (err == cudaSuccess) ? *dm : -1;
  cudaFree(dv);
  cudaFree(dm);
  return m;
}
#endif // _CCCL_CUDA_COMPILATION()

// ---------------------------------------------------------------------------
// Dataset runners: host loop (+ device kernel under CUDA); return total mismatches.
// ---------------------------------------------------------------------------
static int run_f2d(const char* label, const float* vals, int n)
{
  int mism = 0;
  for (int i = 0; i < n; i++)
  {
    if (!f2d_ok(vals[i]))
    {
      ++mism;
    }
  }
#if _CCCL_CUDA_COMPILATION()
  const int dev = device_mismatches<float>(kern_f2d, vals, n);
  mism += (dev < 0) ? 1 : dev;
#endif // _CCCL_CUDA_COMPILATION()
  ::printf("  %-20s: %6d tested, %d mismatches\n", label, n, mism);
  return mism;
}

static int run_d2f(const char* label, const double* vals, int n)
{
  int mism = 0;
  for (int i = 0; i < n; i++)
  {
    if (!d2f_ok(vals[i]))
    {
      ++mism;
    }
  }
#if _CCCL_CUDA_COMPILATION()
  const int dev = device_mismatches<double>(kern_d2f, vals, n);
  mism += (dev < 0) ? 1 : dev;
#endif // _CCCL_CUDA_COMPILATION()
  ::printf("  %-20s: %6d tested, %d mismatches\n", label, n, mism);
  return mism;
}

C2H_TEST("fpemu float->double (widening, exact)", "[fpemu]")
{
  // Every dataset is verified on the host and (under CUDA) on the device.
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  // Boundary values.
  {
    const float vals[] = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      2.0f,
      -2.0f,
      from_f_bits(0x7F800000u), // +Inf
      from_f_bits(0xFF800000u), // -Inf
      from_f_bits(0x7FC00000u), // +qNaN
      from_f_bits(0xFFC00000u), // -qNaN
      from_f_bits(0x7F800001u), // +sNaN
      from_f_bits(0x7FC0DEADu), // qNaN with payload
      from_f_bits(0x00800000u), // min normal float
      from_f_bits(0x7F7FFFFFu), // max finite float
      from_f_bits(0xFF7FFFFFu), // -max finite float
      from_f_bits(0x00000001u), // min positive subnormal
      from_f_bits(0x80000001u), // -min positive subnormal
      from_f_bits(0x007FFFFFu), // max subnormal
      from_f_bits(0x00400000u), // subnormal with 1 bit
      from_f_bits(0x00000100u), // small subnormal
      3.14159265f,
      1.0e38f,
      -1.0e38f,
      1.0e-38f,
      -1.0e-38f,
      1.0e-45f, // near min subnormal
    };
    REQUIRE(run_f2d("boundary", vals, (int) (sizeof(vals) / sizeof(vals[0]))) == 0);
  }

  // Random normal floats.
  {
    constexpr int N = 65536;
    std::vector<float> vals(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0x00800000u, 0x7F7FFFFFu);
    for (int i = 0; i < N; i++)
    {
      uint32_t bits = dist(gen);
      if (gen() & 1)
      {
        bits |= 0x80000000u;
      }
      vals[i] = from_f_bits(bits);
    }
    REQUIRE(run_f2d("random normal", vals.data(), N) == 0);
  }

  // Random subnormal floats.
  {
    constexpr int N = 65536;
    std::vector<float> vals(N);
    std::mt19937 gen(123);
    std::uniform_int_distribution<uint32_t> dist(1u, 0x007FFFFFu);
    for (int i = 0; i < N; i++)
    {
      uint32_t bits = dist(gen);
      if (gen() & 1)
      {
        bits |= 0x80000000u;
      }
      vals[i] = from_f_bits(bits);
    }
    REQUIRE(run_f2d("random subnormal", vals.data(), N) == 0);
  }
}

C2H_TEST("fpemu double->float (narrowing, rounding)", "[fpemu]")
{
  // Every dataset is verified on the host and (under CUDA) on the device.
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  // Boundary values.
  {
    const double vals[] = {
      0.0,
      -0.0,
      1.0,
      -1.0,
      0.5,
      -0.5,
      2.0,
      -2.0,
      from_d_bits(0x7FF0000000000000ULL), // +Inf
      from_d_bits(0xFFF0000000000000ULL), // -Inf
      from_d_bits(0x7FF8000000000000ULL), // +qNaN
      from_d_bits(0xFFF8000000000000ULL), // -qNaN
      from_d_bits(0x7FF0000000000001ULL), // +sNaN
      from_d_bits(0x7FF80000DEADBEEFull), // qNaN with payload
      (double) from_f_bits(0x7F7FFFFFu), // max finite float as double
      -(double) from_f_bits(0x7F7FFFFFu), // -max finite float as double
      (double) from_f_bits(0x00800000u), // min normal float as double
      from_d_bits(0x3800000000000000ULL), // 2^(-127) -> subnormal float
      from_d_bits(0x3690000000000000ULL), // 2^(-150)
      from_d_bits(0x36A0000000000000ULL), // 2^(-149) = min subnormal float
      from_d_bits(0x380FFFFFFFFFE000ULL), // max subnormal float
      1.0e39,
      -1.0e39, // overflow to float Inf
      3.5e38,
      -3.5e38,
      1.0e-300,
      -1.0e-300, // near-zero underflow
      5.0e-324,
      0.25,
      0.125,
      0.0625, // exact conversions
      100.0,
      -100.0,
      3.14159265358979,
    };
    REQUIRE(run_d2f("boundary", vals, (int) (sizeof(vals) / sizeof(vals[0]))) == 0);
  }

  // Random normal doubles in float range.
  {
    constexpr int N = 65536;
    std::vector<double> vals(N);
    std::mt19937 gen(99);
    std::normal_distribution<double> dist(0.0, 1.0e10);
    for (int i = 0; i < N; i++)
    {
      vals[i] = dist(gen);
    }
    REQUIRE(run_d2f("random normal", vals.data(), N) == 0);
  }

  // Random doubles that produce subnormal floats (exponent near -126).
  {
    constexpr int N = 65536;
    std::vector<double> vals(N);
    std::mt19937_64 gen(777);
    for (int i = 0; i < N; i++)
    {
      uint64_t r     = gen();
      int32_t exp_d  = 874 + (int32_t) (r % 23); // biased double exp for subnormal float output
      uint64_t frac  = gen() & 0x000FFFFFFFFFFFFFull;
      uint64_t sign  = (gen() & 1) ? (1ULL << 63) : 0;
      vals[i]        = from_d_bits(sign | ((uint64_t) exp_d << 52) | frac);
    }
    REQUIRE(run_d2f("subnormal output", vals.data(), N) == 0);
  }

  // Rounding tie cases: doubles where the 29 dropped bits = 0x10000000 (half).
  {
    constexpr int N = 8192;
    std::vector<double> vals(N);
    std::mt19937_64 gen(555);
    for (int i = 0; i < N; i++)
    {
      uint64_t r       = gen();
      int32_t exp_d    = 897 + (int32_t) (r % 200);
      uint32_t upper23 = (uint32_t) (r >> 32) & 0x7FFFFF;
      uint64_t frac    = ((uint64_t) upper23 << 29) | (1ULL << 28); // exact halfway point
      uint64_t sign    = (gen() & 1) ? (1ULL << 63) : 0;
      vals[i]          = from_d_bits(sign | ((uint64_t) exp_d << 52) | frac);
    }
    REQUIRE(run_d2f("rounding ties", vals.data(), N) == 0);
  }

  // Overflow region: doubles just around float max.
  {
    constexpr int N = 4096;
    std::vector<double> vals(N);
    std::mt19937_64 gen(333);
    for (int i = 0; i < N; i++)
    {
      uint64_t r    = gen();
      int32_t exp_d = 1149 + (int32_t) (r % 10); // near float overflow
      uint64_t frac = gen() & 0x000FFFFFFFFFFFFFull;
      uint64_t sign = (gen() & 1) ? (1ULL << 63) : 0;
      vals[i]       = from_d_bits(sign | ((uint64_t) exp_d << 52) | frac);
    }
    REQUIRE(run_d2f("overflow region", vals.data(), N) == 0);
  }
}
