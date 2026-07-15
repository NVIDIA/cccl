// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpemu / fpemu_unpacked char + bool operands mirror double.
//
//  bool and character types are excluded from __cccl_is_integer_v, so they are
//  routed through a dedicated converting constructor that upconverts to int32 and
//  reuses the existing int32 path. This mirrors double, for which `1.0 + true` and
//  `1.0 + 'a'` are valid. The same _CCCL_HOST_DEVICE run_test<T>() runs on the host
//  and, under CUDA, on the device via a kernel launch.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// bool and character types must be constructible (mirrors double), while 128-bit
// integers remain deleted.
static_assert(::cuda::std::is_constructible_v<fpemu<double>, bool>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, char>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, signed char>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, unsigned char>, "");
static_assert(::cuda::std::is_constructible_v<fpemu<double>, wchar_t>, "");
static_assert(::cuda::std::is_constructible_v<fpemu_unpacked<double>, bool>, "");
static_assert(::cuda::std::is_constructible_v<fpemu_unpacked<double>, char>, "");
static_assert(::cuda::std::is_constructible_v<fpemu_unpacked<double>, signed char>, "");
static_assert(::cuda::std::is_constructible_v<fpemu_unpacked<double>, unsigned char>, "");

template <class FP>
_CCCL_HOST_DEVICE bool run_test()
{
  const double tol = 1e-10;
  bool ok          = true;

  // Pure construction from bool / char is exact (value <= 255 fits in int32/double).
  ok = ok && ((double) FP(true) == 1.0);
  ok = ok && ((double) FP(false) == 0.0);
  ok = ok && ((double) FP('a') == 97.0);
  ok = ok && ((double) FP((signed char) -5) == -5.0);
  ok = ok && ((double) FP((unsigned char) 200) == 200.0);

  // Mixed arithmetic mirrors double: 1.0 + true + 'a' == 1 + 1 + 97 == 99.
  {
    const double ref = 1.0 + true + 'a';
    FP a(1.0);
    FP r = FP(a + true) + 'a';
    ok   = ok && (::cuda::std::fabs((double) r - ref) <= tol);
  }

  // char on the left-hand side of a mixed op.
  {
    const double ref = 'a' + 2.0;
    FP b(2.0);
    ok = ok && (::cuda::std::fabs((double) FP('a' + b) - ref) <= tol);
  }

  // bool used as a multiplicative mask (true -> keep, false -> zero).
  {
    FP c(3.5);
    ok = ok && (::cuda::std::fabs((double) FP(c * true) - 3.5) <= tol);
    ok = ok && (::cuda::std::fabs((double) FP(c * false) - 0.0) <= tol);
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
template <class FP>
__global__ void run_test_kernel(bool* out)
{
  *out = run_test<FP>();
}

template <class FP>
static bool device_ok()
{
  bool* d_ok = nullptr;
  if (cudaMallocManaged(&d_ok, sizeof(bool)) != cudaSuccess)
  {
    return false;
  }
  *d_ok = false;
  run_test_kernel<FP><<<1, 1>>>(d_ok);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
  {
    err = cudaDeviceSynchronize();
  }
  const bool ok = (err == cudaSuccess) && *d_ok;
  cudaFree(d_ok);
  return ok;
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu char/bool mixed ops mirror double", "[fpemu]")
{
  fp_ran_on_host();
  REQUIRE(run_test<fpemu<double>>());
  REQUIRE(run_test<fpemu_unpacked<double>>());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  REQUIRE(device_ok<fpemu<double>>());
  REQUIRE(device_ok<fpemu_unpacked<double>>());
#endif // _CCCL_CUDA_COMPILATION()
}
