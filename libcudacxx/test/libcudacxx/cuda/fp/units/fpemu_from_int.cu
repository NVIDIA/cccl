// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: integer -> fp64emu -> double conversions (bit-exact).
//
//  Validates that fpemu integer-to-double conversions produce bit-identical
//  results to native casts for int32_t / uint32_t / int64_t / uint64_t as well as
//  long long / unsigned long long (which route through the constrained integer
//  constructor template). int32/uint32 are always exact; the 64-bit types require
//  round-to-nearest-even for values with more than 53 significant bits. The same
//  _CCCL_HOST_DEVICE per-value check runs on the host over each dataset and, under
//  CUDA, on the device via a mismatch-counting kernel.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_HAS_INT128()
// 128-bit integer construction is deliberately deleted: it would silently truncate
// to 64 bits. Verify no emulated type is constructible from __int128 while the
// standard integer widths remain constructible.
static_assert(!::cuda::std::is_constructible_v<fpemu<double>, __int128_t>, "");
static_assert(!::cuda::std::is_constructible_v<fpemu<double>, __uint128_t>, "");
static_assert(!::cuda::std::is_constructible_v<fpemu_unpacked<double>, __int128_t>, "");
static_assert(!::cuda::std::is_constructible_v<fpemu_unpacked<double>, __uint128_t>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, int64_t>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, uint64_t>, "");
#endif // _CCCL_HAS_INT128()

// Convert one integer through fp64emu and compare bit-for-bit against the native
// cast to double.
template <class T>
_CCCL_HOST_DEVICE bool int_ok(T v)
{
  fp64emu e(v);
  return ::cuda::std::bit_cast<uint64_t>((double) e) == ::cuda::std::bit_cast<uint64_t>((double) v);
}

#if _CCCL_CUDA_COMPILATION()
template <class T>
__global__ void kern_int(const T* v, int n, int* mism)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (!int_ok(v[i]))
    {
      atomicAdd(mism, 1);
    }
  }
}

// Returns device mismatch count, or -1 on a CUDA API failure.
template <class T>
static int device_mismatches(const T* vals, int n)
{
  T* dv   = nullptr;
  int* dm = nullptr;
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
  kern_int<T><<<blocks, threads>>>(dv, n, dm);
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

// Host loop (+ device kernel under CUDA); returns total mismatches.
template <class T>
static int run_int(const char* label, const T* vals, int n)
{
  int mism = 0;
  for (int i = 0; i < n; i++)
  {
    if (!int_ok(vals[i]))
    {
      ++mism;
    }
  }
#if _CCCL_CUDA_COMPILATION()
  const int dev = device_mismatches<T>(vals, n);
  mism += (dev < 0) ? 1 : dev;
#endif // _CCCL_CUDA_COMPILATION()
  ::printf("  %-10s: %6d tested, %d mismatches\n", label, n, mism);
  return mism;
}

static const int32_t int32_vals[] = {
  0,
  1,
  -1,
  2,
  -2,
  100,
  -100,
  1000000,
  -1000000,
  INT32_MAX,
  INT32_MIN,
  INT32_MIN + 1,
  INT32_MAX - 1,
  0x7FFFFFFF,
  (int32_t) 0x80000000,
  12345678,
  -12345678,
};

static const uint32_t uint32_vals[] = {
  0,
  1,
  2,
  100,
  1000000,
  0x7FFFFFFFu,
  0x80000000u,
  0xFFFFFFFFu,
  0xFFFFFFFEu,
  42,
  999999999,
  0x12345678u,
  0xDEADBEEFu,
};

static const int64_t int64_vals[] = {
  0,
  1,
  -1,
  2,
  -2,
  INT32_MAX,
  INT32_MIN,
  (int64_t) INT32_MAX + 1,
  (int64_t) INT32_MIN - 1,
  INT64_MAX,
  INT64_MIN,
  INT64_MIN + 1,
  INT64_MAX - 1,
  (1LL << 53),
  -(1LL << 53),
  (1LL << 53) + 1,
  (1LL << 53) - 1,
  (1LL << 53) + 2,
  (1LL << 53) + 3,
  (1LL << 54) - 1,
  -(1LL << 54) + 1,
  (1LL << 62),
  -(1LL << 62),
  0x100000000LL,
  -0x100000000LL,
  123456789012345LL,
  -123456789012345LL,
};

static const uint64_t uint64_vals[] = {
  0,
  1,
  2,
  (uint64_t) UINT32_MAX,
  (uint64_t) UINT32_MAX + 1,
  UINT64_MAX,
  UINT64_MAX - 1,
  (1ULL << 53),
  (1ULL << 53) + 1,
  (1ULL << 53) - 1,
  (1ULL << 53) + 2,
  (1ULL << 53) + 3,
  (1ULL << 54) - 1,
  (1ULL << 63),
  (1ULL << 63) + 1,
  0x8000000000000000ULL,
  0xFFFFFFFF00000000ULL,
  123456789012345ULL,
  9999999999999999ULL,
};

static const long long longlong_vals[] = {
  0,
  1,
  -1,
  2,
  -2,
  INT32_MAX,
  INT32_MIN,
  (long long) INT32_MAX + 1,
  (long long) INT32_MIN - 1,
  INT64_MAX,
  INT64_MIN,
  INT64_MIN + 1,
  INT64_MAX - 1,
  (1LL << 53),
  -(1LL << 53),
  (1LL << 53) + 1,
  (1LL << 53) - 1,
  (1LL << 62),
  -(1LL << 62),
  123456789012345LL,
  -123456789012345LL,
};

static const unsigned long long ulonglong_vals[] = {
  0,
  1,
  2,
  (unsigned long long) UINT32_MAX,
  (unsigned long long) UINT32_MAX + 1,
  UINT64_MAX,
  UINT64_MAX - 1,
  (1ULL << 53),
  (1ULL << 53) + 1,
  (1ULL << 53) - 1,
  (1ULL << 63),
  0x8000000000000000ULL,
  0xFFFFFFFF00000000ULL,
  123456789012345ULL,
  9999999999999999ULL,
};

template <class T, int N>
static constexpr int count_of(const T (&)[N])
{
  return N;
}

C2H_TEST("fpemu int->double (bit-exact)", "[fpemu]")
{
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  REQUIRE(run_int("int32_t", int32_vals, count_of(int32_vals)) == 0);
  REQUIRE(run_int("uint32_t", uint32_vals, count_of(uint32_vals)) == 0);
  REQUIRE(run_int("int64_t", int64_vals, count_of(int64_vals)) == 0);
  REQUIRE(run_int("uint64_t", uint64_vals, count_of(uint64_vals)) == 0);
  REQUIRE(run_int("longlong", longlong_vals, count_of(longlong_vals)) == 0);
  REQUIRE(run_int("ulonglong", ulonglong_vals, count_of(ulonglong_vals)) == 0);
}
